"""Talks to whichever answer-generating LLM is configured.

Why a provider abstraction? Ollama only exists on the machine it's installed
on — a Streamlit Cloud deployment can never reach "localhost:11434" on your
laptop. So a deployed copy of DocChat needs a hosted API instead. This module
looks up config, in order:

  1. The app owner's Streamlit secrets (st.secrets) — set these in the app's
     "Settings > Secrets" when deployed on Streamlit Community Cloud.
  2. Plain environment variables — populated locally from a `.env` file
     (see `.env.example`) via python-dotenv, or set however you like in CI/
     other hosts.
  3. A local Ollama server, for people running DocChat entirely on their own
     machine with no API key at all.

Embeddings are handled separately (see app/ingest.py / app/rag.py) using
Chroma's bundled local embedding model, so uploading a PDF never depends on
any of this.
"""
import os

import httpx
from dotenv import load_dotenv

load_dotenv()  # populate os.environ from a local .env file, if present

try:
    import streamlit as st
except ImportError:  # running under the FastAPI backend, not Streamlit
    st = None

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}

# Shown in the UI so people know where to get a free key.
PROVIDER_INFO = {
    "anthropic": {"label": "Anthropic (Claude)", "key_url": "https://console.anthropic.com/settings/keys"},
    "groq": {"label": "Groq (free tier)", "key_url": "https://console.groq.com/keys"},
    "openai": {"label": "OpenAI", "key_url": "https://platform.openai.com/api-keys"},
    "ollama": {"label": "Ollama (local only)", "key_url": None},
}


def _get_config(key: str) -> str | None:
    """Look up a setting: st.secrets (deployed) > environment variable (.env locally)."""
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key)


def active_provider() -> str:
    """Which provider will actually be used right now."""
    forced = _get_config("LLM_PROVIDER")
    if forced:
        return forced.lower()
    if _get_config("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _get_config("GROQ_API_KEY"):
        return "groq"
    if _get_config("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def is_ready() -> bool:
    """True if the active provider has everything it needs to be called."""
    provider = active_provider()
    if provider == "ollama":
        return True  # only knowable by actually trying to connect
    return bool(_get_config(f"{provider.upper()}_API_KEY"))


def _chat_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate_anthropic(system_prompt: str, user_prompt: str) -> str:
    api_key = _get_config("ANTHROPIC_API_KEY")
    model = _get_config("ANTHROPIC_MODEL") or DEFAULT_MODELS["anthropic"]
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    blocks = response.json().get("content", [])
    return "".join(block.get("text", "") for block in blocks)


def _generate_openai_compatible(base_url: str, api_key: str, model: str,
                                 system_prompt: str, user_prompt: str) -> str:
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": _chat_messages(system_prompt, user_prompt)},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _generate_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = _get_config("GROQ_API_KEY")
    model = _get_config("GROQ_MODEL") or DEFAULT_MODELS["groq"]
    return _generate_openai_compatible(
        "https://api.groq.com/openai/v1", api_key, model, system_prompt, user_prompt
    )


def _generate_openai(system_prompt: str, user_prompt: str) -> str:
    api_key = _get_config("OPENAI_API_KEY")
    model = _get_config("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
    return _generate_openai_compatible(
        "https://api.openai.com/v1", api_key, model, system_prompt, user_prompt
    )


def _generate_ollama(system_prompt: str, user_prompt: str) -> str:
    model = _get_config("OLLAMA_CHAT_MODEL") or DEFAULT_MODELS["ollama"]
    try:
        response = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": _chat_messages(system_prompt, user_prompt),
                "stream": False,
            },
            timeout=httpx.Timeout(120, connect=5),
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as e:
        raise RuntimeError(
            "No LLM is configured. Ollama only runs on your own computer, so a "
            "deployed copy of this app can't reach it. Set GROQ_API_KEY (or "
            "another provider's key) in .env locally, or in the app's Streamlit "
            "secrets when deployed."
        ) from e


_PROVIDERS = {
    "anthropic": _generate_anthropic,
    "groq": _generate_groq,
    "openai": _generate_openai,
    "ollama": _generate_ollama,
}


def generate(system_prompt: str, user_prompt: str) -> str:
    """Generate an answer from whichever LLM provider is active.

    Why? This is the "G" in RAG — the LLM uses the retrieved context + system
    rules to write a grounded answer.
    """
    provider = active_provider()
    generate_fn = _PROVIDERS.get(provider)
    if generate_fn is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(_PROVIDERS)}."
        )

    if provider != "ollama" and not _get_config(f"{provider.upper()}_API_KEY"):
        info = PROVIDER_INFO[provider]
        raise RuntimeError(
            f"No API key set for {info['label']}. Set {provider.upper()}_API_KEY "
            f"in .env locally, or in the app's Streamlit secrets when deployed "
            f"(get a free key at {info['key_url']})."
        )

    label = PROVIDER_INFO.get(provider, {}).get("label", provider)
    try:
        return generate_fn(system_prompt, user_prompt)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"{label} API error ({e.response.status_code}): {e.response.text[:200]}"
        ) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Could not reach {label}: {e}") from e

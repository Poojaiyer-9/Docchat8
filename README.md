# DocChat - Voice RAG

A privacy-first, voice-capable document Q&A assistant with a Streamlit interface. Embeddings and the vector store always run locally (no API key, no external service); the answer-writing LLM is pluggable, so the same app runs equally well on your laptop with Ollama or deployed for anyone to use with a hosted API key.

## ✨ Features
- 📄 Upload PDFs and ask questions in natural language
- 🎤 Voice input (browser-native, no uploads needed)
- 🔊 Voice output (text-to-speech answers)
- 👤 Persona-aware answers (Student / Researcher / Executive)
- 📌 Traceable citations with page numbers
- 🔒 Local embeddings — your document text never has to leave the box running Chroma
- 🔌 Bring-your-own-LLM: Ollama, Groq, Anthropic (Claude), or OpenAI
- 🎨 A distinctive "reading room" dark UI built with Streamlit

## 🏗️ Architecture
```
User → Streamlit UI → [Ingestion: PDF → Chunks → Chroma's local embedding model]
       ↓
    [Query: Chroma search → Retrieve top-K chunks]
       ↓
    [Persona-aware prompt → LLM provider → Answer + Citations + Audio]
```

## ⚠️ Why this matters if you deploy it (e.g. Streamlit Community Cloud)

Ollama is a program that runs on **your own machine**. A deployed copy of this
app runs on someone else's server, which can never reach `localhost:11434` on
your laptop — trying to do so used to fail with a raw socket error
(`[Errno 99] Cannot assign requested address`) as soon as a visitor uploaded a
PDF. This app now:
- Embeds documents with Chroma's own small local model (downloaded once, no
  API key, works identically on your laptop or in the cloud).
- Talks to the answer-writing LLM through a pluggable client
  (`app/llm_client.py`) that supports hosted APIs — so anyone visiting the
  deployed app can paste their own key into the sidebar and use it
  immediately, with no server-side setup from you.
- Only tries to reach Ollama when you've deliberately picked it as the
  provider, and fails with an actionable message instead of a raw socket
  error if it's unreachable.

## 🚀 Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose an LLM provider

**Option A — a hosted API (works anywhere, no local install):**
Get a free key from one of:
- [Groq](https://console.groq.com/keys) — free tier, fast (recommended default)
- [Anthropic](https://console.anthropic.com/settings/keys) — Claude models
- [OpenAI](https://platform.openai.com/api-keys)

You can paste the key straight into the app's sidebar at runtime — nothing to
configure up front. To pre-fill it for your own deployment instead, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
one key.

**Option B — Ollama, fully local:**
```bash
ollama pull llama3.2
ollama serve
```
Then pick "Ollama (localhost only)" in the sidebar. This only works when
you're running the app yourself on the same machine as the Ollama server —
it can't work for a public deployment.

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
The app will automatically open at `http://localhost:8501`

## 📦 Tech Stack
- **Frontend:** Streamlit
- **Embeddings:** Chroma's bundled local MiniLM model (no API key required)
- **LLM:** Groq / Anthropic / OpenAI / Ollama — pick one in the sidebar
- **Vector DB:** Chroma (local)
- **Voice:** Web Speech API (input) + gTTS (output)
- **PDF Processing:** PyMuPDF

## 🎯 How It Works

1. **Upload a PDF** — DocChat chunks the document and stores embeddings in Chroma
2. **Choose a persona** — Student (simple), Researcher (technical), or Executive (bullets)
3. **Ask questions** — Type or use voice input
4. **Get cited answers** — Responses include source page numbers and audio playback

## 🔑 Key Innovations

| Feature | What existing tools miss | How DocChat fixes it |
|---------|--------------------------|---------------------|
| **Local embeddings** | ChatPDF/Quivr upload docs to a cloud embedding service | Chunking + embedding always run locally, no key needed |
| **Persona-aware** | All tools give generic answers | Switches complexity (simple/technical/bullets) |
| **Voice I/O** | Rare in RAG tools | Speak questions, get spoken answers |
| **Traceable citations** | Vague "source" labels | Exact page number + chunk snippet |
| **Runs for anyone** | Local-only tools break once deployed | Any visitor can paste their own free LLM key in the sidebar |

## 📝 Project Structure
```
docchat/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI server (legacy)
│   ├── rag.py           # Core RAG logic
│   ├── ingest.py        # PDF processing + Chroma storage (local embeddings)
│   ├── llm_client.py    # Pluggable LLM client (Groq / Anthropic / OpenAI / Ollama)
│   └── voice.py         # Speech-to-text + text-to-speech
├── .streamlit/
│   └── secrets.toml.example  # Copy to secrets.toml to pre-fill an API key
├── static/              # Audio files and assets
├── chroma_db/           # Local vector database
├── app.py               # Streamlit UI (main entry point)
├── requirements.txt
└── README.md
```

## 🎓 Perfect for Your Resume

This project demonstrates:
- **End-to-end RAG pipeline** (chunking → embeddings → vector search → LLM generation)
- **Local LLM integration** with Ollama (no cloud dependencies)
- **Vector database** usage with Chroma
- **Voice-capable interfaces** (Web Speech API + gTTS)
- **Persona-aware prompting** (role-conditioned generation)
- **Modern UI development** with Streamlit
- **API design** and async programming

## 🙋‍♀️ Author
Built by Pooja V — AIML undergrad passionate about practical AI systems.

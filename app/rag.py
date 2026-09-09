from app.llm_client import generate
import chromadb
from chromadb.utils import embedding_functions

# Initialize local Chroma vector DB (same DB as ingest.py — shares the ./chroma_db folder)
client = chromadb.PersistentClient(path="./chroma_db")
# Must match the embedding function ingest.py used to build the collection.
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name="docs", embedding_function=embedding_function)

# Define personas: each has a system prompt that changes answer style
# Why? No need to retrain the model — just change the instructions!
PERSONAS = {
    "student": "You are a helpful tutor for a high-school/early college student. "
               "Explain concepts simply with analogies, avoid jargon, keep answers short. "
               "Use ONLY the context below. If the answer is not in context, say 'I don't know'. "
               "Cite the source page number for every claim.",

    "researcher": "You are a research assistant. Be precise, technical, and thorough. "
                  "Include methodology details, data, and exact citations with page numbers. "
                  "Use ONLY the context below. If the answer is not in context, say 'I don't know'.",

    "executive": "You are an assistant for a business executive. Answer in 3 bullet points max. "
                 "Focus only on key numbers, action items, and high-level insights. "
                 "Use ONLY the context below. If the answer is not in context, say 'I don't know'."
}


def retrieve(question: str, top_k: int = 3) -> tuple[list[str], list[dict]]:
    """Retrieve the top-K most relevant chunks for a question, with metadata.

    Why? This is the "R" in RAG — we fetch the exact parts of the document that
    answer the user's question, so the LLM doesn't have to guess.
    """
    # Search Chroma for the top-K most similar chunks (Chroma embeds the
    # question itself, using the same embedding_function as the collection)
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    # Extract the chunk text and their metadata (source, page, etc.)
    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    return chunks, metadata


def format_citations(metadata: list[dict]) -> list[dict]:
    """Format metadata into clean citation objects for the user.

    Why? Users need to verify where the answer came from. We return the exact
    page and source so they can check the PDF themselves.
    """
    citations = []
    seen = set()  # Avoid duplicate citations
    for m in metadata:
        key = (m["source"], m["page"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": m["source"],
                "page": m["page"],
                "chunk_id": m.get("chunk_id", "")
            })
    return citations


def ask(question: str, persona: str = "student", top_k: int = 3) -> dict:
    """Main RAG pipeline: retrieve → generate → return answer with citations.

    This is the core function that ties everything together.
    """
    # 1. Retrieve relevant chunks
    chunks, metadata = retrieve(question, top_k)

    # 2. Combine chunks into a single context string
    context = "\n\n".join(chunks)

    # 3. Select system prompt based on user's chosen persona
    system_prompt = PERSONAS.get(persona, PERSONAS["student"])

    # 4. Build the full prompt: context + question
    user_prompt = f"""
CONTEXT FROM DOCUMENT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using the context above. If the answer is not present, say "I don't know".
2. Cite the source page number for every claim you make.
3. Follow the persona rules in the system prompt.
"""

    # 5. Generate answer from local LLM
    answer = generate(system_prompt, user_prompt)

    # 6. Format citations
    citations = format_citations(metadata)

    return {
        "answer": answer,
        "citations": citations,
        "persona": persona
    }

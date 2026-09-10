import os
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from pymupdf import open as pdf_open

# Use a writable runtime directory. The checked-in repository directory can be
# read-only on Streamlit Cloud, while both modules still share this path.
chroma_path = os.environ.get(
    "CHROMA_DB_PATH", os.path.join(tempfile.gettempdir(), "docchat_chroma_db")
)
client = chromadb.PersistentClient(path=chroma_path)

# Chroma's bundled local embedding model (small ONNX MiniLM, downloaded once
# from the public internet on first use). Runs on CPU, needs no API key and
# no Ollama, so uploading a PDF works the same on a laptop or in the cloud.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Get or create a collection named "docs" (like a table in SQL)
collection = client.get_or_create_collection(name="docs", embedding_function=embedding_function)


def clear_collection():
    """Delete all documents from the collection.
    
    Why? When a user uploads a new PDF, we want to start fresh instead of
    mixing old and new document chunks. This ensures answers come only from
    the currently uploaded document.
    """
    # Keep the collection object alive because rag.py holds its own handle to it.
    # Deleting and recreating the collection leaves that handle pointing at a
    # collection ID that no longer exists.
    existing = collection.get()
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def get_indexed_docs() -> list[dict]:
    """Get list of all indexed documents with their metadata.
    
    Returns list of dicts with 'source' and 'page' keys.
    """
    try:
        # Get all documents (limit to 1000 for safety)
        results = collection.get(limit=1000)
        if not results or "metadatas" not in results or not results["metadatas"]:
            return []
        
        # Extract unique documents
        docs = []
        seen = set()
        for metadata in results["metadatas"]:
            if metadata:
                source = metadata.get("source", "unknown")
                if source not in seen:
                    seen.add(source)
                    docs.append({"source": source})
        return docs
    except Exception:
        return []


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split long text into small overlapping chunks.

    Why? LLMs can't read a 100-page PDF at once (limited context window).
    Chunks are small searchable units. Overlap ensures we don't cut sentences
    in half and lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap  # Move forward, keep overlap
    return chunks


def ingest_pdf(file_path: str, filename: str) -> int:
    """Process a PDF: extract text, chunk, embed, store in Chroma.

    Returns number of chunks created.
    """
    # Open PDF with PyMuPDF
    doc = pdf_open(file_path)
    all_chunks = []
    all_metadata = []
    all_ids = []

    # Iterate over every page in the PDF
    for page_num, page in enumerate(doc):
        text = page.get_text()  # Extract text from page
        if not text.strip():
            continue  # Skip empty pages

        # Split page text into chunks
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            # Store metadata: source filename, page number, chunk ID
            all_metadata.append({
                "source": filename,
                "page": page_num + 1,  # Pages are 1-indexed for users
                "chunk_id": f"{filename}_p{page_num + 1}_c{i}"
            })
            all_ids.append(f"{filename}_p{page_num + 1}_c{i}")

    # Store in Chroma: it embeds `documents` itself using `embedding_function`
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadata
    )
    return len(all_chunks)

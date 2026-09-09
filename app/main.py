from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.rag import ask
from app.voice import text_to_speech
import os
import uuid

app = FastAPI(title="DocChat - Local Voice RAG")

# Serve static files (HTML, audio files)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Data Models (validates user input) ---
class QuestionRequest(BaseModel):
    question: str
    persona: str = "student"  # Default to student mode
    top_k: int = 3


# --- Endpoints ---
@app.get("/")
def serve_ui():
    """Serve the chat UI when user visits the root URL."""
    return FileResponse("static/index.html")


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF to index it for Q&A.

    Why? Users need to add their documents before asking questions.
    """
    # Save uploaded file temporarily (use uuid to avoid collisions)
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        # Read contents and explicitly close the upload handle to avoid file lock
        contents = file.file.read()
        file.file.close()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Ingest the PDF (chunk, embed, store)
        num_chunks = ingest_pdf(temp_path, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": num_chunks,
            "message": f"Indexed {file.filename} with {num_chunks} chunks. You can now ask questions!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Delete temp file after processing (ignore if still locked by OS)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                pass  # File may be temporarily locked; will be cleaned on restart


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """Ask a question about uploaded documents.

    Returns answer + citations + optional audio.
    """
    # Run RAG pipeline
    result = ask(req.question, req.persona, req.top_k)

    # Generate audio for the answer (optional, for voice output)
    audio_path = text_to_speech(result["answer"])
    result["audio_url"] = f"/static/{os.path.basename(audio_path)}"

    return result

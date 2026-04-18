import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_document
from app.query import ask_llm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/chat")
def chat_ui():
    return FileResponse("app/static/index.html")

# Allow frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/upload")
async def upload_document(
    tenant_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and ingest a PDF document for a tenant"""

    # Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into ChromaDB
    result = ingest_document(file_path, tenant_id)

    return {"tenant_id": tenant_id, "file": file.filename, **result}


@app.post("/ask")
async def ask_question(
    tenant_id: str = Form(...),
    question: str = Form(...)
):
    """Ask a question against a tenant's documents"""

    # Get answer from Ollama
    result = ask_llm(question, tenant_id)

    return {
        "tenant_id": tenant_id,
        "question": question,
        "answer": result["answer"],
        "chunks_found": result["chunks_found"]
    }
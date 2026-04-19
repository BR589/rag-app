import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_document
from app.query import ask_llm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/chat")
def chat_ui():
    return FileResponse("app/static/index.html")

UPLOAD_DIR = "./uploads"
HISTORY_FILE = "./conversation_history.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/upload")
async def upload_document(
    tenant_id: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_document(file_path, tenant_id)
    return {"tenant_id": tenant_id, "file": file.filename, **result}


@app.post("/ask")
async def ask_question(
    tenant_id: str = Form(...),
    question: str = Form(...)
):
    all_history = load_history()
    history = all_history.get(tenant_id, [])

    result = ask_llm(question, tenant_id, history)

    if tenant_id not in all_history:
        all_history[tenant_id] = []

    all_history[tenant_id].append({
        "user": question,
        "assistant": result["answer"]
    })

    all_history[tenant_id] = all_history[tenant_id][-10:]
    save_history(all_history)

    return {
        "tenant_id": tenant_id,
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "chunks_found": result["chunks_found"]
    }


@app.post("/clear-history")
async def clear_history(tenant_id: str = Form(...)):
    all_history = load_history()
    all_history[tenant_id] = []
    save_history(all_history)
    return {"status": "cleared"}
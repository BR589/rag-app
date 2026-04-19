import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_document
from app.query import ask_llm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.query import ask_llm, get_collection, groq_client

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
@app.post("/summarize")
async def summarize_documents(tenant_id: str = Form(...)):
    """Auto summarize all uploaded documents for a tenant"""
    try:
        collection = get_collection(tenant_id)
        results = collection.get(include=["documents", "metadatas"])

        if not results["documents"]:
            return {"summary": "No documents found. Please upload documents first."}

        # Group by filename
        files = {}
        for i, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][i] or {}
            filename = metadata.get("filename", "unknown")
            if filename not in files:
                files[filename] = []
            files[filename].append(doc)

        # Build summary prompt
        all_content = ""
        for filename, chunks in files.items():
            sample = " ".join(chunks[:5])  # First 5 chunks per file
            all_content += f"\n\nDocument: {filename}\n{sample}"

        prompt = f"""Summarize the following documents in a structured way.
For each document provide:
- Document name
- Main topic (1 sentence)
- Key points (3-5 bullet points)
- Important terms or concepts mentioned

Documents:
{all_content}

Provide a clear, concise summary:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )

        return {"summary": response.choices[0].message.content}

    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}"}


@app.post("/clear-history")
async def clear_history(tenant_id: str = Form(...)):
    all_history = load_history()
    all_history[tenant_id] = []
    save_history(all_history)
    return {"status": "cleared"}
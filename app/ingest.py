import os
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

# Lazy load
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

client = chromadb.PersistentClient(path="./chroma_db")


def get_collection(tenant_id: str):
    """Each tenant gets their own isolated collection"""
    return client.get_or_create_collection(name=f"tenant_{tenant_id}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_document(file_path: str, tenant_id: str):
    """Parse PDF, chunk it, embed it, store in ChromaDB"""

    # Step 1: Extract text from PDF
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    if not text.strip():
        return {"error": "Could not extract text from PDF"}

    # Step 2: Chunk the text
    chunks = chunk_text(text)

    # Step 3: Generate embeddings
    embeddings = get_model().encode(chunks).tolist()

    # Step 4: Store in ChromaDB under tenant's collection
    collection = get_collection(tenant_id)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    )

    return {"status": "success", "chunks_stored": len(chunks)}
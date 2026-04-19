import os
import pypdf
import chromadb
from fastembed import TextEmbedding

# Lightweight embedding model (~50MB vs 500MB before)
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="/tmp/chroma_db")


def get_collection(tenant_id: str):
    return client.get_or_create_collection(name=f"tenant_{tenant_id}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_document(file_path: str, tenant_id: str):
    # Extract text
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    if not text.strip():
        return {"error": "Could not extract text from PDF"}

    # Chunk
    chunks = chunk_text(text)

    # Embed using fastembed
    embeddings = list(embedding_model.embed(chunks))
    embeddings = [e.tolist() for e in embeddings]

    # Store
    collection = get_collection(tenant_id)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    )

    return {"status": "success", "chunks_stored": len(chunks)}
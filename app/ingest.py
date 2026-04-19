import os
import re
import pypdf
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="/tmp/chroma_db")


def get_collection(tenant_id: str):
    return client.get_or_create_collection(
        name=f"tenant_{tenant_id}",
        embedding_function=ef
    )


def extract_text_by_page(file_path: str):
    """Extract text page by page, preserving page numbers"""
    pages = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_by_paragraph(pages: list, max_chunk_size: int = 400, overlap_sentences: int = 2):
    """
    Split text into meaningful chunks by paragraph.
    Keeps sentences together, adds overlap between chunks.
    Tracks source page number for citations.
    """
    chunks = []

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                word_count = len(sentence.split())
                
                if current_size + word_count > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num
                    })
                    # Keep last few sentences as overlap
                    current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
                    current_size = sum(len(s.split()) for s in current_chunk)

                current_chunk.append(sentence)
                current_size += word_count

        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "page": page_num
            })

    return chunks


def ingest_document(file_path: str, tenant_id: str):
    """Parse PDF, chunk smartly, store with metadata"""

    # Extract text with page numbers
    pages = extract_text_by_page(file_path)

    if not pages:
        return {"error": "Could not extract text from PDF"}

    # Smart chunking
    chunks = chunk_by_paragraph(pages)

    if not chunks:
        return {"error": "No content found after chunking"}

    # Prepare for ChromaDB
    collection = get_collection(tenant_id)
    filename = os.path.basename(file_path)

    documents = [c["text"] for c in chunks]
    ids = [f"{filename}_p{c['page']}_chunk_{i}" for i, c in enumerate(chunks)]
    metadatas = [{"filename": filename, "page": c["page"]} for c in chunks]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "status": "success",
        "chunks_stored": len(chunks),
        "pages_processed": len(pages)
    }
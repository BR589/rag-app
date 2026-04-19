import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="/tmp/chroma_db")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_collection(tenant_id: str):
    return client.get_or_create_collection(
        name=f"tenant_{tenant_id}",
        embedding_function=ef
    )


def retrieve_chunks(question: str, tenant_id: str, top_k: int = 10):
    """Retrieve top K most relevant chunks with metadata"""
    collection = get_collection(tenant_id)
    
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] or {}
        distance = results["distances"][0][i]
        chunks.append({
            "text": doc,
            "filename": metadata.get("filename", "unknown"),
            "page": metadata.get("page", "?"),
            "relevance": round(1 - distance, 3)
        })

    return chunks


def build_prompt(question: str, chunks: list, history: list):
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} — {chunk['filename']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    history_str = ""
    if history:
        history_str = "\n\n--- CONVERSATION HISTORY ---\n"
        for turn in history[-6:]:
            history_str += f"User asked: {turn['user']}\nYou answered: {turn['assistant']}\n\n"
        history_str += "--- END OF HISTORY ---\n"

    prompt = f"""You are a precise document assistant. Follow these rules:
1. Answer in 2-4 sentences unless more detail is needed
2. If the user says "elaborate", "explain more", "tell me more", or similar — expand on YOUR LAST ANSWER from conversation history
3. Only cite the specific source that directly answers — maximum 2 citations
4. If answer is not in sources AND not in history, say: "This information is not available in the uploaded documents."
5. NEVER say a question is too vague — always try to answer using history context
{history_str}
--- DOCUMENT SOURCES ---
{context}

User Question: {question}

Answer:"""

    return prompt

def ask_llm(question: str, tenant_id: str, history: list = []):
    """Retrieve chunks and get answer with citations"""

    chunks = retrieve_chunks(question, tenant_id)

    if not chunks:
        return {
            "answer": "No documents found. Please upload a document first.",
            "chunks_found": 0,
            "citations": []
        }

    prompt = build_prompt(question, chunks, history)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Lower = more factual, less creative
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    # Only return top 2 most relevant citations
    citations = [
        {
            "filename": c["filename"],
            "page": c["page"],
            "relevance": c["relevance"]
        }
        for c in chunks[:2] if c["relevance"] > 0.3
    ]

    return {
        "answer": answer,
        "chunks_found": len(chunks),
        "citations": citations
    }
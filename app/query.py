import chromadb
from fastembed import TextEmbedding
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Same lightweight model as ingest.py
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="/tmp/chroma_db")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_collection(tenant_id: str):
    return client.get_or_create_collection(name=f"tenant_{tenant_id}")


def retrieve_chunks(question: str, tenant_id: str, top_k: int = 5):
    # Embed the question
    question_embedding = list(embedding_model.embed([question]))[0].tolist()

    # Search ChromaDB
    collection = get_collection(tenant_id)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = results["documents"][0]
    return chunks


def build_prompt(question: str, chunks: list):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask_llm(question: str, tenant_id: str):
    chunks = retrieve_chunks(question, tenant_id)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "chunks_found": len(chunks)}
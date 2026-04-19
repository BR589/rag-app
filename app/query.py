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


def retrieve_chunks(question: str, tenant_id: str, top_k: int = 5):
    collection = get_collection(tenant_id)
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results["documents"][0]


def build_prompt(question: str, chunks: list):
    context = "\n\n".join(chunks)
    return f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


def ask_llm(question: str, tenant_id: str):
    chunks = retrieve_chunks(question, tenant_id)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "chunks_found": len(chunks)}
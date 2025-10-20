import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client (new architecture)
chroma_client = chromadb.PersistentClient(path="./embeddings")
collection = chroma_client.get_or_create_collection(name="doc_chunks")

def ask_question(query, top_k=5):
    print(f"\n[?] Question: {query}")

    # Embed the query
    query_embedding = embedder.encode(query).tolist()

    # Retrieve top-k relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract documents
    retrieved_chunks = results['documents'][0]
    sources = results['metadatas'][0]

    # 🔍 Debug: Print retrieved chunks
    print("\n[🔍] Retrieved Chunks:\n")
    for chunk in retrieved_chunks:
        print(chunk)

    # Combine chunks into context
    context = "\n\n".join(retrieved_chunks)

    # Send to Ollama
    prompt = f"""You are a helpful assistant. Based on the following context, answer the question below.

Context:
{context}

Question:
{query}
"""
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    answer = response['message']['content']

    print("\n[✓] Answer:")
    print(answer)
    print("\n[📄] Sources:")
    for src in sources:
        print(f"- {src['source']}")

if __name__ == "__main__":
    user_query = input("Ask a question about your documents: ")
    ask_question(user_query)
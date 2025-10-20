import os
import chromadb
from sentence_transformers import SentenceTransformer
from utils.file_loader import load_document
from utils.chunker import chunk_text


# Initialize ChromaDB client (new architecture)
chroma_client = chromadb.PersistentClient(path="./embeddings")
collection = chroma_client.get_or_create_collection(name="doc_chunks")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Scan docs folder
DOCS_DIR = "./docs"
for filename in os.listdir(DOCS_DIR):
    file_path = os.path.join(DOCS_DIR, filename)
    try:
        print(f"[+] Processing: {filename}")
        text = load_document(file_path)

        if not text.strip():
            print(f"[!] Skipped empty file: {filename}")
            continue

        chunks = chunk_text(text)
        print(f"[🔍] Found {len(chunks)} chunks in {filename}")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{i}"
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{"source": filename}]
            )
        print(f"[✓] Embedded {len(chunks)} chunks from {filename}")
    except Exception as e:
        print(f"[!] Failed to process {filename}: {e}")

print("[✓] All documents embedded and saved.")
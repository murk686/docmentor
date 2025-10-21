# embedder.py
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./embeddings")
collection = chroma_client.get_or_create_collection(name="doc_chunks")
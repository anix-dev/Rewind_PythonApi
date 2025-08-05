# app/db/chroma_client.py

from chromadb import PersistentClient
import os

# Update this path as needed
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

try:
    chroma_client = PersistentClient(path=CHROMA_DB_DIR)
    print("✅ ChromaDB initialized using new PersistentClient.")
except Exception as e:
    print(f"❌ Failed to initialize ChromaDB: {e}")
    chroma_client = None

# app/db/embedding_model.py

from sentence_transformers import SentenceTransformer

# Use this for production-grade models (lightweight and fast)
MODEL_NAME = "all-MiniLM-L6-v2"

try:
    embedder = SentenceTransformer(MODEL_NAME)
    print(f"✅ Embedding model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    embedder = None
    print(f"❌ Failed to load embedding model: {e}")


def get_embedding(text: str) -> list[float]:
    if embedder is None:
        raise RuntimeError("Embedding model is not initialized.")
    return embedder.encode(text).tolist()

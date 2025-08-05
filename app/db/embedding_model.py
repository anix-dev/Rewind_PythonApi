# app/db/embedding_model.py

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# For compatibility, provide an alias if other parts of the codebase expect 'embedder'
embedder = embed_model

print("✅ HuggingFace embed_model initialized.")

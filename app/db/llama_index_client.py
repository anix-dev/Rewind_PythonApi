import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from chromadb import PersistentClient

# GROQ/Gemini LLM imports
USE_GROQ = bool(os.getenv("GROQ_API_KEY"))
USE_GEMINI = bool(os.getenv("GEMINI_API_KEY"))

if USE_GROQ:
    from llama_index.llms.groq import Groq
    llm = Groq(
        model="llama3-70b-8192",  # or mixtral, gemma-7b-it
        api_key=os.getenv("GROQ_API_KEY"),
    )
elif USE_GEMINI:
    from llama_index.llms.gemini import Gemini
    llm = Gemini(api_key=os.getenv("GEMINI_API_KEY"))
else:
    raise Exception("❌ No GROQ_API_KEY or GEMINI_API_KEY found in environment.")

# Embeddings: use local or OSS (since GROQ doesn’t offer embeddings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Settings for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# Chroma setup
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
collection_name = os.getenv("CHROMA_COLLECTION", "rewind-ai")

chroma_client = PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=collection)

# Build the index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print(f"✅ LlamaIndex initialized using {'GROQ' if USE_GROQ else 'Gemini'} with ChromaDB.")

import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from chromadb import PersistentClient

# Flags for LLM selection
USE_GROQ = bool(os.getenv("GROQ_API_KEY"))
USE_GEMINI = bool(os.getenv("GEMINI_API_KEY"))

# LLM setup
if USE_GROQ:
    from llama_index.llms.groq import Groq
    llm = Groq(
        model="llama3-70b-8192",  # or mixtral, gemma-7b-it
        api_key=os.getenv("GROQ_API_KEY"),
    )
    print("✅ Using Groq: llama3-70b-8192 pricing as per Groq API rates.")

elif USE_GEMINI:
    from llama_index.llms.gemini import Gemini
    llm = Gemini(
        model="gemini-2.5-flash-lite",   # ✅ Explicit model selection for cheapest tier
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    print("✅ Using Gemini 2.5 Flash-Lite at $0.10/M input tokens & $0.40/M output tokens.")

else:
    raise Exception("❌ No GROQ_API_KEY or GEMINI_API_KEY found in environment.")

# Local embeddings (free)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Apply settings globally
Settings.llm = llm
Settings.embed_model = embed_model

# ChromaDB setup (local)
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
collection_name = os.getenv("CHROMA_COLLECTION", "rewind-ai")

chroma_client = PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=collection)

# Build the index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print(f"✅ LlamaIndex initialized using {'GROQ' if USE_GROQ else 'Gemini'} with ChromaDB.")

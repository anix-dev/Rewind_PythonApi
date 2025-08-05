# app/db/llama_index_client.py

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore  # ✅ correct after upgrade
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from chromadb import PersistentClient
import os

# Path where ChromaDB stores vectors
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# Chroma client
chroma_client = PersistentClient(path=CHROMA_DB_DIR)

# Create or connect to a collection
collection = chroma_client.get_or_create_collection("rewind-ai")

# Wrap it in a LlamaIndex vector store
chroma_vector_store = ChromaVectorStore(chroma_collection=collection)

# ✅ Define embedding model and LLM explicitly
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo")

# ✅ Set global settings (replaces ServiceContext)
Settings.embed_model = embed_model
Settings.llm = llm

# Build the index (empty or can load docs later)
index = VectorStoreIndex.from_vector_store(
    vector_store=chroma_vector_store
)

print("✅ LlamaIndex initialized with ChromaDB.")

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from chromadb import PersistentClient
import os

# Load environment config
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# Load ChromaDB persistent client
chroma_client = PersistentClient(path=CHROMA_DB_DIR)
chroma_collection = chroma_client.get_or_create_collection("rewind_index")

# Setup LlamaIndex
Settings.embed_model = OpenAIEmbedding()
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index once
index = VectorStoreIndex.from_documents([], storage_context=storage_context)


async def index_user_data(user_id: str, moods: list[dict], replays: list[dict]):
    try:
        documents = []

        # Moods
        for mood in moods:
            text = mood.get("user_text") or mood.get("ai_response")
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "user_id": str(mood["user"]),
                            "type": "mood",
                            "id": str(mood["_id"]),
                        }
                    )
                )

        # Replays
        for replay in replays:
            text = replay.get("gem_response") or replay.get("user_response")
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "user_id": str(replay["user"]),
                            "type": "replay",
                            "id": str(replay["_id"]),
                        }
                    )
                )

        print(f"📥 Indexing data for user {user_id}...")
        if documents:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            print(f"✅ Successfully indexed {len(documents)} documents for user {user_id}")
        else:
            print(f"ℹ️ No documents to index for user {user_id}")

    except Exception as e:
        print(f"❌ Failed to index user data: {e}")
        raise

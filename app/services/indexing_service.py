# app/services/indexing_service.py

from typing import List
from llama_index.core.schema import Document
from app.db.llama_index_client import index  # Ensure llama_index_client.py exports 'index'

def format_for_indexing(user_id: str, moods: list, replays: list) -> List[Document]:
    """
    Convert moods and replays into LlamaIndex-compatible Document objects.
    """
    documents = []

    for mood in moods:
        location_str = None
        if mood.get("latitude") and mood.get("longitude"):
            location_str = f"{mood['latitude']},{mood['longitude']}"

        doc = Document(
            text=mood.get("user_text", ""),
            metadata={
                "type": "mood",
                "user_id": user_id,
                "mood": mood.get("mood"),
                "location": location_str,
                "date": str(mood.get("create_date")),
                "source_id": str(mood.get("_id")),
            }
        )
        documents.append(doc)

    for replay in replays:
        doc = Document(
            text=replay.get("gem_response", ""),
            metadata={
                "type": "replay",
                "user_id": user_id,
                "location": replay.get("location") or None,  # already a string
                "date": str(replay.get("create_date")),
                "source_id": str(replay.get("_id")),
            }
        )
        documents.append(doc)

    return documents

async def index_user_data(user_id: str, moods: list, replays: list):
    """
    Index the user's moods and replays into the vector database.
    """
    try:
        print(f"📥 Indexing data for user {user_id}...")

        docs = format_for_indexing(user_id, moods, replays)

        if not docs:
            print(f"⚠️ No documents to index for user {user_id}")
            return

        # This replaces the current reference documents with new ones
        index.refresh_ref_docs(docs)
        print(f"✅ Successfully indexed {len(docs)} documents for user {user_id}")

    except Exception as e:
        print(f"❌ Failed to index user data: {e}")
        raise e

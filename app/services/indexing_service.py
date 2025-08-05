# app/services/indexing_service.py

from typing import List
from llama_index.core.schema import Document
from app.db.llama_index_client import index  # Ensure llama_index_client.py exports 'index'

from datetime import datetime


def format_for_indexing(user_id: str, moods: list, replays: list) -> List[Document]:
    """
    Convert moods and replays into LlamaIndex-compatible Document objects.
    """
    documents = []

    for mood in moods:
        try:
            location_str = None
            if mood.get("latitude") and mood.get("longitude"):
                location_str = f"{mood['latitude']},{mood['longitude']}"

            # Safely format date
            create_date = mood.get("create_date")
            formatted_date = create_date.isoformat() if isinstance(create_date, datetime) else str(create_date)

            # Flatten context_tags list to comma-separated string
            context_tags = mood.get("context_tags", [])
            context_tag_str = ", ".join(context_tags) if isinstance(context_tags, list) else str(context_tags)

            doc = Document(
                text=mood.get("user_text", ""),
                metadata={
                    "type": "mood",
                    "user_id": user_id,
                    "mood": mood.get("mood"),
                    "ai_response": mood.get("ai_response", ""),
                    "location": location_str,
                    "date": formatted_date,
                    "context_tags": context_tag_str,
                    "replay_opportunity_score": mood.get("replay_opportunity_score"),
                    "source_id": str(mood.get("_id")),
                }
            )
            documents.append(doc)
        except Exception as e:
            print(f"⚠️ Failed to format mood document: {e}")

    for replay in replays:
        try:
            # Safely format date
            create_date = replay.get("create_date")
            formatted_date = create_date.isoformat() if isinstance(create_date, datetime) else str(create_date)

            # Flatten context_tags list to comma-separated string
            context_tags = replay.get("context_tags", [])
            context_tag_str = ", ".join(context_tags) if isinstance(context_tags, list) else str(context_tags)

            doc = Document(
                text=replay.get("gem_response", ""),
                metadata={
                    "type": "replay",
                    "user_id": user_id,
                    "user_response": replay.get("user_response", ""),
                    "location": replay.get("location", None),
                    "date": formatted_date,
                    "context_tags": context_tag_str,
                    "replay_opportunity_score": replay.get("replay_opportunity_score"),
                    "mood_ref_id": str(replay.get("moods")),
                    "source_id": str(replay.get("_id")),
                }
            )
            documents.append(doc)
        except Exception as e:
            print(f"⚠️ Failed to format replay document: {e}")

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

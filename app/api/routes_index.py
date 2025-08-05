from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from typing import List

from bson import ObjectId

from app.db.mongo_client import db
from app.db.llama_index_client import index  # Your shared LlamaIndex instance
from app.services.indexing_service import index_user_data

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

router = APIRouter()

# --- Models ---

class SearchRequest(BaseModel):
    user_id: str
    query: str

class IndexRequest(BaseModel):
    user_id: str

# --- Helpers ---

def serialize_mongo_doc(doc):
    doc["_id"] = str(doc["_id"])
    if "user" in doc and isinstance(doc["user"], ObjectId):
        doc["user"] = str(doc["user"])
    return doc

# --- Routes ---

@router.post("/index-user-data")
async def index_user(request: Request):
    """
    Index a user's moods and replays into ChromaDB via LlamaIndex.
    """
    body = await request.json()
    user_id = body.get("user_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    moods = await db.moods.find({"user": object_id}).sort("create_date", -1).to_list(100)
    replays = await db.replays.find({"user": object_id}).sort("create_date", -1).to_list(100)

    print(f"📝 Indexing {len(moods)} moods and {len(replays)} replays for user {user_id}")

    try:
        await index_user_data(user_id, moods, replays)
        return {
            "status": "ok",
            "message": f"Indexed {len(moods)} moods and {len(replays)} replays",
            "total": len(moods) + len(replays)
        }
    except Exception as e:
        print(f"❌ Failed to index user data: {e}")
        raise HTTPException(status_code=500, detail="Failed to index user data")


@router.post("/search-memories")
async def search_memories(request: SearchRequest):
    """
    Semantic search on indexed memories using user's query.
    """
    try:
        print("🧠 Query received:", request.query)

        filters = MetadataFilters(filters=[MetadataFilter(key="user_id", value=request.user_id)])
        query_engine = index.as_query_engine(similarity_top_k=3, filters=filters)
        response = query_engine.query(request.query)

        print("🔍 Raw LLM Response:", response)

        if hasattr(response, 'source_nodes'):
            print("📚 Retrieved Docs:")
            for node in response.source_nodes:
                print("  👉", node.node.text[:100], "...")

        return {
            "result": str(response) if response and str(response).strip() else "🤖 I couldn't find anything relevant."
        }

    except Exception as e:
        print("❌ Error during memory search:", e)
        return {"error": "Internal Server Error", "details": str(e)}


@router.get("/user-indexed-data")
async def get_indexed_data(user_id: str = Query(...), top_k: int = 5):
    """
    Fetch top indexed documents for a user (for debugging or UI preview).
    """
    try:
        filters = MetadataFilters(filters=[MetadataFilter(key="user_id", value=user_id)])
        query_engine = index.as_query_engine(similarity_top_k=top_k, filters=filters)
        response = query_engine.query("dummy")  # 'dummy' forces vector retrieval

        if not hasattr(response, 'source_nodes'):
            return {"message": "No indexed data found."}

        documents: List[str] = []
        for node in response.source_nodes:
            text = node.node.text.strip()
            documents.append(text[:1000])  # Clip long results

        return {
            "user_id": user_id,
            "documents": documents
        }

    except Exception as e:
        print("❌ Error fetching indexed data:", e)
        return {"error": "Failed to fetch user data", "details": str(e)}

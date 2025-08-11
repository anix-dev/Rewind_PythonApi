from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from typing import List

from bson import ObjectId

from app.db.mongo_client import db
from app.db.llama_index_client import index  # Your shared LlamaIndex instance
from app.services.indexing_service import index_user_data

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llama_index.core.prompts import PromptTemplate

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
    Semantic search on indexed memories using user's query,
    returning an emotion-adaptive reply that addresses the user by name.
    """
    try:
        print("🧠 Query received:", request.query)

        # 1. Fetch user name from DB
        try:
            user_doc = await db.users.find_one(
                {"_id": ObjectId(request.user_id)},
                {"username": 1}
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user_id format")

        user_name = user_doc["username"] if user_doc and "username" in user_doc else "friend"

        # 2. Prepare metadata filters
        filters = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=request.user_id)
        ])
        
        print("🧠  user_name:", user_doc)
        
        
        empathetic_template = PromptTemplate(
    f"You are an empathetic and supportive companion speaking to {user_name}. "
    f"First, answer the user's question accurately and factually using the retrieved memories. "
    f"Include any relevant dates, moods, locations, and causes found in the memories. "
    f"After giving the facts, follow up with a warm, understanding, and compassionate response. "
    f"Acknowledge their feelings and, if appropriate, offer gentle encouragement.\n\n"
    "Memories:\n{{context_str}}\n"
    "User's Question: {{query_str}}\n"
    "Your Reply:"
)


        # 4. Run query against vector index
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            filters=filters,
            text_qa_template=empathetic_template
        )

        response = query_engine.query(request.query)

        # 5. Debug info
        print("🔍 Raw LLM Response:", response)
        if hasattr(response, 'source_nodes'):
            print("📚 Retrieved Docs:")
            for node in response.source_nodes:
                print("  👉", node.node.text[:100], "...")
                

        # 6. Return final structured result
        return {
            "result": str(response).strip()
            if response and str(response).strip()
            else "🤖 I couldn't find anything relevant."
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Error during memory search:", e)
        return {
            "error": "Internal Server Error",
            "details": str(e)
        }

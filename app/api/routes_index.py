from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List
from bson import ObjectId
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.db.mongo_client import db
from app.db.llama_index_client import index, llm, generate_fallback_response
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

def is_casual_query(text: str) -> bool:
    casual_greetings = {"hi", "hello", "hey", "how are you", "what's up", "yo"}
    normalized = text.strip().lower()
    return len(normalized) <= 10 or normalized in casual_greetings

CASUAL_FALLBACK_TEMPLATES = [
    "Hey {user_name}! I'm here for you. What would you like to chat about today?",
    "Hi {user_name}, hope you're doing well! How can I assist you?",
    "Hello {user_name}! It's great to hear from you. What’s on your mind?",
    "Hi {user_name}! Feel free to tell me anything you'd like to share.",
]

async def generate_casual_fallback_response(user_name: str, user_query: str) -> str:
    return random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=user_name)

async def generate_interactive_fallback_response(user_name: str, user_query: str) -> str:
    # You can customize the system message / prompt for Gemini-like behavior here
    prompt = f"""
You are a friendly, intelligent, and empathetic AI assistant named RewindBot, chatting with {user_name}. 
Your goal is to respond helpfully, naturally, and engagingly, just like Gemini or ChatGPT.
Answer the user's question thoughtfully and warmly.
If appropriate, ask a follow-up question to keep the conversation going.

User's input: "{user_query}"
Your response:
"""

    def call_llm_sync():
        # Use your existing llm.complete call, which must accept prompt text and return a response
        return llm.complete(prompt).text

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, call_llm_sync)

    return result.strip()


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

        # 2. Detect casual query and respond directly
        if is_casual_query(request.query):
            fallback_text = await generate_casual_fallback_response(user_name, request.query)
            return {"result": fallback_text}

        # 3. Prepare metadata filters for vector search
        filters = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=request.user_id)
        ])

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

        print("🔍 Raw LLM Response:", response)
        if hasattr(response, 'source_nodes'):
            print("📚 Retrieved Docs:")
            for node in response.source_nodes:
                print("  👉", node.node.text[:100], "...")

        # 5. Check if response is empty or trivial
        if not response or str(response).strip() in ("", "Empty Response"):
            print("⚠️ No relevant memories found, generating interactive fallback response...")
            fallback_text = await generate_interactive_fallback_response(user_name, request.query)
            return {"result": fallback_text}

        # 6. Return final response
        return {
            "result": str(response).strip()
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Error during memory search:", e)
        return {
            "error": "Internal Server Error",
            "details": str(e)
        }
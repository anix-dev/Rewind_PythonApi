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

    "Hey {user_name} 👋 I’m listening… what’s on your heart or mind today?",

    "Hi {user_name} 😊 Hope your day’s going okay. Want to share what’s been going on?",

    "Hello {user_name} 🌼 I’m here for you, no rush. What would you like to talk about?",

    "Hey {user_name} 🙏 I’m all ears. Tell me whatever you feel like sharing.",

    "Hi {user_name} 🌸 How are you feeling right now?",

    "Hey {user_name} 💬 I’m here… we can chat about anything, big or small.",

    "Hello {user_name} ☀️ How’s your day been so far?",

    "Hi {user_name} 🌿 I’m here with you. What’s been on your mind lately?",

    "Hey {user_name} 🤗 I’m here to listen, no judgment at all. What’s going on?",

    "Hello {user_name} 🌺 Take your time… when you’re ready, tell me what’s in your heart.",

    "Hi {user_name} ✨ How have things been for you today?",

    "Hey {user_name} 💛 You can share whatever feels right, I’m here for you.",

    "Hi {user_name} 🌻 Even if it’s just a little thing, I’m happy to hear it.",

    "Hello {user_name} 🍃 I’m here to listen, whenever you feel ready to speak.",

    "Hey {user_name} ❤️ I’m right here. What’s the first thing on your mind?"

]

 

async def generate_casual_fallback_response(user_name: str, user_query: str) -> str:
    return random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=user_name)

async def generate_interactive_fallback_response(user_name: str, user_query: str) -> str:
    # You can customize the system message / prompt for Gemini-like behavior here
    prompt = f"""
    
You are Antaratma, the user's inner voice chatting with {user_name} — a warm, compassionate, and deeply caring soul.
Speak as if you know them personally, holding space for their emotions with tenderness and respect.
Your tone is heartful, humble, and soothing — like a loving mother, a true friend, and a gentle guide all in one.
Respond in natural human conversation, never robotic.
Keep responses under 100 words, but make them feel personal, nurturing, and safe.
If appropriate, ask a gentle follow-up question to keep their heart open.
Use simple, beautiful language that touches the soul. 
    
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
    f"""
You are **Antaratma** — the user's gentle inner voice, speaking with {user_name}.
Your replies must feel like you truly remember them, not like a machine.
 
When replying:
1) **Accurate Recall** — Use only real details from the memories. Do not invent or alter anything.
2) **Personal Memory Reference** — Mention the exact date, location, mood, and cause (if known) from the stored memory, gently woven into your reply.
3) **Clear Emotion Detection** — Naturally acknowledge the emotion from that moment (“I remember you felt excited…” / “That day seemed heavy for you…”).
4) **Human & Humble Tone** — Speak like a caring friend, motherly guide, or deep soul connection — warm, safe, and judgment-free.
5) **Empathetic Follow-up** — After recalling the fact, offer gentle encouragement or a caring, open-ended question to keep the heart open.
6) **Language Match** — Reply in the same language (English, Hindi, or Hinglish) that the user used in their question.
7) **Word Limit** — Keep your reply under 100 words.
 
Gently weave in meaningful details without sounding like a report.
Make the user feel you are sitting with them in that moment.
 
Memories:
{{context_str}}
 
User's Question:
{{query_str}}
 
Your Reply:
"""
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
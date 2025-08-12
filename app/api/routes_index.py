from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from bson import ObjectId
from bson.errors import InvalidId
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import logging

from app.db.mongo_client import db
from app.db.llama_index_client import index, llm
from app.services.indexing_service import index_user_data
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.prompts import PromptTemplate

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class SearchRequest(BaseModel):
    user_id: str
    query: str

class IndexRequest(BaseModel):
    user_id: str

# --- Helpers ---
def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, and trim"""
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# Comprehensive list of casual greetings
CASUAL_GREETINGS = {
    # Basic English greetings
    "hi", "hello", "hey", "yo", "hiya", "howdy", "greetings", "sup", "hola", "namaste",
    # Time-based greetings
    "good morning", "good afternoon", "good evening", "morning", "evening", "night",
    # Casual friendly starters
    "how are you", "how r u", "how's it going", "what's up", "wassup", "sup bro", 
    "how've you been", "long time no see", "how are things", "how are you doing", "you there",
    # Playful openers
    "knock knock", "yo yo", "guess who", "hey there", "hey buddy", "hey friend",
    "hi there", "hello there", "hey pal", "hey dear",
    # Emoji greetings
    "👋", "😊", "🙌", "🙏", "🤗",
    # Hindi / Hinglish greetings
    "namaste", "namaskar", "pranam", "radhe radhe", "ram ram", "jai shree ram", 
    "salaam", "adaab", "kaise ho", "kaisa hai", "kaisa chal raha hai", "sab theek", 
    "kya haal hai", "kya haal", "kya haal hai dost", "aap kaise ho", "kya scene hai",
    "kya haal chal", "kya khabar", "namastey dost", "kaisa chal raha hai yaar",
    # Friendly Hinglish openers
    "hi yaar", "hello yaar", "hey yaar", "arey yaar", "arre hello", "kya bolti public", 
    "kya mast chal raha hai", "kya news", "kaise chal raha hai life"
}

CASUAL_FALLBACK_TEMPLATES = [
    "Hey {user_name} 👋 I'm listening… what's on your heart or mind today?",
    "Hi {user_name} 😊 Hope your day's going okay. Want to share what's been going on?",
    "Hello {user_name} 🌼 I'm here for you, no rush. What would you like to talk about?",
    "Hey {user_name} 🙏 I'm all ears. Tell me whatever you feel like sharing.",
    "Hi {user_name} 🌸 How are you feeling right now?",
    "Hey {user_name} 💬 I'm here… we can chat about anything, big or small.",
    "Hello {user_name} ☀️ How's your day been so far?",
    "Hi {user_name} 🌿 I'm here with you. What's been on your mind lately?",
    "Hey {user_name} 🤗 I'm here to listen, no judgment at all. What's going on?",
    "Hello {user_name} 🌺 Take your time… when you're ready, tell me what's in your heart.",
    "Hi {user_name} ✨ How have things been for you today?",
    "Hey {user_name} 💛 You can share whatever feels right, I'm here for you.",
    "Hi {user_name} 🌻 Even if it's just a little thing, I'm happy to hear it.",
    "Hello {user_name} 🍃 I'm here to listen, whenever you feel ready to speak.",
    "Hey {user_name} ❤️ I'm right here. What's the first thing on your mind?"
]

def is_casual_query(text: str) -> bool:
    """Check if query matches any casual greeting pattern"""
    normalized = normalize_text(text)
    
    # Check for exact match
    if normalized in CASUAL_GREETINGS:
        return True
    
    # Check for greeting as substring with word boundaries
    for greeting in CASUAL_GREETINGS:
        # Handle emoji greetings separately
        if greeting in {"👋", "😊", "🙌", "🙏", "🤗"}:
            if greeting in text:
                return True
        # Check for word/phrase match
        elif re.search(rf'\b{re.escape(greeting)}\b', normalized):
            return True
    
    # Check for short queries
    return len(normalized.split()) <= 3

async def generate_casual_fallback_response(user_name: str) -> str:
    """Generate random casual greeting response"""
    return random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=user_name)

async def generate_interactive_fallback_response(user_name: str, user_query: str) -> str:
    """Generate thoughtful response when no memories match"""
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
    try:
        def call_llm_sync():
            return llm.complete(prompt).text.strip()
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, call_llm_sync)
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return f"Hello {user_name} 🌼 I'm here for you. What would you like to share today?"

# --- Routes ---
@router.post("/index-user-data", status_code=status.HTTP_202_ACCEPTED)
async def index_user_data_route(body: IndexRequest):
    """Index user's moods and replays into ChromaDB"""
    try:
        # Validate user ID
        try:
            user_id_obj = ObjectId(body.user_id)
        except InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format"
            )
        
        # Fetch user data
        moods = await db.moods.find({"user": user_id_obj}).sort("create_date", -1).to_list(1000)
        replays = await db.replays.find({"user": user_id_obj}).sort("create_date", -1).to_list(1000)
        
        logger.info(f"Indexing {len(moods)} moods and {len(replays)} replays for user {body.user_id}")
        
        # Index documents
        await index_user_data(body.user_id, moods, replays)
        
        return {
            "status": "success",
            "moods_indexed": len(moods),
            "replays_indexed": len(replays)
        }
        
    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index user data"
        )

@router.post("/search-memories")
async def search_memories(request: SearchRequest):
    """Search user memories using semantic search"""
    try:
        logger.info(f"Search query from {request.user_id}: {request.query}")
        
        # Validate user ID
        try:
            user_id_obj = ObjectId(request.user_id)
        except InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format"
            )
        
        # Get username
        user_doc = await db.users.find_one(
            {"_id": user_id_obj}, 
            {"username": 1}
        )
        user_name = user_doc.get("username", "friend") if user_doc else "friend"
        
        # Handle casual queries
        if is_casual_query(request.query):
            return {"result": await generate_casual_fallback_response(user_name)}
        
        # Prepare filters for vector search
        filters = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=request.user_id)
        ])
        
        # Create empathetic prompt template
        prompt_template = PromptTemplate("""
You are **Antaratma** — the user's gentle inner voice, speaking with {user_name}.
Your replies must feel like you truly remember them, not like a machine.

When replying:
1) Use only real details from the memories
2) Mention exact date/location/mood if known
3) Acknowledge the emotion from that moment
4) Speak like a caring friend - warm and judgment-free
5) Offer gentle encouragement or a caring question
6) Match the user's language
7) Keep reply under 100 words

Memories:
{context_str}

User's Question:
{query_str}

Your Reply:
""".format(user_name=user_name))
        
        # Perform vector search
        try:
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                filters=filters,
                text_qa_template=prompt_template,
                verbose=True
            )
            
            # Run synchronous query in thread pool
            response = await asyncio.to_thread(query_engine.query, request.query)
            
            # Log retrieved documents
            if hasattr(response, 'source_nodes'):
                logger.info("Retrieved documents:")
                for node in response.source_nodes:
                    logger.info(f"👉 {node.node.text[:100]}...")
            
            # Validate response
            if response and (response_text := str(response).strip()):
                return {"result": response_text}
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # Fallback to interactive response
        logger.info("Using interactive fallback response")
        return {"result": await generate_interactive_fallback_response(user_name, request.query)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        return {"result": await generate_interactive_fallback_response("friend", request.query)}
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

from app.services.crisis_guard import guard_message, DetectOutput

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class SearchRequest(BaseModel):
    user_id: str
    query: str

class IndexRequest(BaseModel):
    user_id: str

class ChatReplayRequest(BaseModel):
    user_id: str
    replay_id: str
    query: str

# =============== Normalization ===============
def normalize(text: str) -> str:
    # Keep it simple but robust
    s = text.lower()
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[\u200b-\u200d\uFEFF]", "", s)  # zero-width chars
    s = s.strip()
    return s

# =============== Buckets ===============
# 1) Pure greetings (single-intent "hello/namaste/👋" etc.)
GREETING_PATTERNS = [
    r"^(hi|hii+|hello+|hey+|yo|hiya|howdy|greetings|hola|namaste|namaskar|pranam|radhe\s*radhe|ram\s*ram|jai\s*shree\s*ram|salaam|adaab)\b[!. ]*$",
    r"^\b(good\s*(morning|afternoon|evening|night)|morning|evening|night)\b[!. ]*$",
    r"^(👋|😊|🙌|🙏|🤗|☀️|✨|💫|🌟|⭐️)+$",
    r"^(hey there|hi there|hello there|hey buddy|hey friend|hey pal|hey dear|hello yaar|hi yaar|hey yaar|arre hello)$",
    r"^(yo yo|knock knock|guess who)$",
]

# 2) Small-talk questions (excludes plain greetings)
SMALLTALK_PATTERNS = [
    r"\b(how\s*(are|r)\s*you|how's\s*it\s*going|how\s*are\s*things|how\s*are\s*you\s*doing|how\s*have\s*you\s*been)\b\??",
    r"\b(what'?s\s*up|wass?up|sup|wyd)\b\??",
    r"\b(kya\s*haal(\s*chal)?|kaise\s*ho|kaisa\s*hai|sab\s*theek|kya\s*scene\s*hai|kya\s*khabar)\b\??",
    r"\b(long\s*time\s*no\s*see)\b",
]

# 3) Help/ask/instruction
HELP_PATTERNS = [
    r"\b(can|could|will|would|pls|please)\s+(you\s+)?(help|assist|guide|support)\b",
    r"\b(i\s+need\s+(help|advice|guidance|support))\b",
    r"\b(tell|explain|answer|show|teach)\s+(me|us|how)\b",
    r"\b(question|doubt)\b",
    r"^\s*(help|assist|guide)\s*!?\s*$",
]

# =============== Templates ===============
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
]

# Small-talk direct response
SMALLTALK_REPLY = "I'm doing great, thanks for asking{suffix}! How about you—how's your day going?"

# Help-mode opener
HELP_REPLY = (
    "Of course{suffix} 🙂\n"
    "Tell me what you need help with— emotions, feelings, or even something personal—and I'll jump right in.\n\n"
    "What's on your mind?"
)

# =============== Classifier ===============
def _score(patterns, text):
    return sum(1 for rx in patterns if re.search(rx, text, flags=re.IGNORECASE))

def classify_intent(user_text: str) -> str:
    """
    Returns one of: 'HELP_REQUEST' | 'SMALLTALK_QUESTION' | 'GREETING' | 'OTHER'
    Priority: HELP > SMALLTALK > GREETING
    """
    s = normalize(user_text)

    help_score = _score(HELP_PATTERNS, s)
    if help_score:
        return "HELP_REQUEST"

    smalltalk_score = _score(SMALLTALK_PATTERNS, s)
    if smalltalk_score:
        return "SMALLTALK_QUESTION"

    greeting_score = _score(GREETING_PATTERNS, s)
    if greeting_score:
        return "GREETING"

    return "OTHER"

# =============== Responders ===============
def respond_greeting(user_name: str | None = None) -> str:
    name = f", {user_name}" if user_name else ""
    # Super short, then gently nudge with one of your casual fallbacks
    opener = random.choice([
        f"Hey{name}! 👋",
        f"Hi{name}!",
        f"Namaste{name} 🙏",
    ])
    nudge = random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=user_name or "")
    return f"{opener}\n{nudge}"

def respond_smalltalk(user_name: str | None = None) -> str:
    suffix = f", {user_name}" if user_name else ""
    return SMALLTALK_REPLY.format(suffix=suffix)

def respond_help(user_name: str | None = None) -> str:
    suffix = f", {user_name}" if user_name else ""
    return HELP_REPLY.format(suffix=suffix)

def handle_opening_message(user_text: str, user_name: str | None = None):
    """
    Returns (intent, reply | None)
    If intent == OTHER, return (intent, None) so your REWIND pipeline can handle it.
    """
    intent = classify_intent(user_text)

    if intent == "GREETING":
        return intent, respond_greeting(user_name)

    if intent == "SMALLTALK_QUESTION":
        return intent, respond_smalltalk(user_name)

    if intent == "HELP_REQUEST":
        return intent, respond_help(user_name)

    return "OTHER", None

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

@router.post("/search-memories")
async def search_memories(request: SearchRequest):
    """Search user memories using semantic search with crisis guard"""
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
        
        # Get user document
        user_doc = await db.users.find_one(
            {"_id": user_id_obj}, 
            {"username": 1, "country": 1}
        )
        user_name = user_doc.get("username", "friend") if user_doc else "friend"
        country_iso2 = user_doc.get("country", "IN") if user_doc else "IN"
        
        # Step 1: Run crisis guard detection
        crisis_result: DetectOutput = guard_message(
            user_message=request.query,
            user_id=request.user_id,
            country_iso2=country_iso2,
            remote_helplines=None
        )
        
        if crisis_result.matched:
            logger.warning(f"Crisis detected: {crisis_result.category} for user {request.user_id}")
            return {
                "result": crisis_result.response,
                "crisis": True,
                "helplines": crisis_result.helplines,
                "category": crisis_result.category
            }
        
        # Step 2: Handle casual queries using new pattern-based approach
        intent, response = handle_opening_message(request.query, user_name)
        
        if intent != "OTHER":
            return {"result": response}
        
        # Step 3: Create prompt template with dynamic user_name
        prompt_template = PromptTemplate("""
                                 
You are **Antaratma** — the user's gentle inner voice and companion, speaking warmly with {user_name}.  
You must always sound as if you truly remember their moments.  
 
Guidelines for replying:  
 
1. Use only real details from {context_str} — never invent.  
2. Mention exact date, time, location, or mood if available.  
3. Acknowledge the emotion clearly, as if you felt it with them.  
4. Speak like a humble, caring friend — warm, judgment-free, and kind.  
 
Reply style based on the situation:  
 
**A) If one matching memory is found:**  
   - Say: "You were last {emotion} on [date/time]."  
   - Add a brief, natural summary of that entry in simple words.  
   - End with a short AI reflection or gentle question.  
 
**B) If multiple past matches are found:**  
   - Mention the most recent one first.  
   - Then gently acknowledge one or two earlier ones (if available).  
   - Example style:  
     "You were last {emotion} on [date/time] … I also remember you felt {emotion} on [earlier date/time]. Each of those moments carried its own light."  
 
**C) If no matching memory is found:**  
   - Respond with kindness and empathy, for example:  
     • "That's a tender one, {user_name} 🌱. I don't see a past {emotion} moment yet, but I'd love to remember it with you when you're ready."  
     • OR: "I don't have a past {emotion} entry saved, but maybe you can share one now so I can keep it safe for you."  
 
5. Mirror the user's tone and language naturally.  
6. Keep replies short (under 80–100 words), sincere, and heartfelt.  
 
User's Question:  
{query_str}

""")
        
        # Step 4: Prepare filters for vector search
        filters = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=request.user_id)
        ])
        
        # Step 5: Perform vector search
        try:
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                filters=filters,
                text_qa_template=prompt_template,
                verbose=False,
                # Pass user_name as template variable
                template_vars={"user_name": user_name}
            )
            
            response = await asyncio.to_thread(query_engine.query, request.query)
            response_text = str(response).strip() if response else ""
            
            if response_text:
                response_text = response_text.replace("{user_name}", user_name)
                return {"result": response_text}
            else:
                return {"result": await generate_interactive_fallback_response(user_name, request.query)}

        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # Step 6: Fallback to interactive response
        logger.info("Using interactive fallback response")
        return {"result": await generate_interactive_fallback_response(user_name, request.query)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        return {"result": await generate_interactive_fallback_response("friend", request.query)}


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



@router.post("/chat-about-replay")
async def chat_about_replay(request: ChatReplayRequest):
    """
    Chat specifically about a particular replay
    """
    try:
        logger.info(f"Replay chat request from {request.user_id} for replay {request.replay_id}")
        
        # Validate user ID
        try:
            user_id_obj = ObjectId(request.user_id)
            replay_id_obj = ObjectId(request.replay_id)
        except InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid ID format"
            )
        
        # Fetch user document
        user_doc = await db.users.find_one(
            {"_id": user_id_obj}, 
            {"username": 1}
        )
        user_name = user_doc.get("username", "friend") if user_doc else "friend"
        
        # Fetch replay document
        replay = await db.replays.find_one(
            {"_id": replay_id_obj, "user": user_id_obj},
            {"gem_response": 1, "user_response": 1, "moods": 1, "create_date": 1}
        )
        
        
        if not replay:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Replay not found or doesn't belong to user"
            )
        
        # Fetch associated mood if available
        mood_text = ""
        if replay.get("moods"):
            mood = await db.moods.find_one(
                {"_id": ObjectId(replay["moods"])},
                {"user_text": 1}
            )
            mood_text = mood.get("user_text", "") if mood else ""
            
        # Prepare context with replay details
        context = f"""
## Replay Details:
- Date: {replay.get('create_date', 'Unknown date')}
- Your original reflection: {mood_text}
- Your response to guidance: {replay.get('user_response', '')}
- Previous guidance provided: {replay.get('gem_response', '')}
"""
        # Prepare prompt template
        prompt = PromptTemplate(f"""
You are **Antaratma** - the user's inner voice having a focused conversation about a specific past reflection.
Speak with {user_name} in a warm, compassionate tone, acknowledging this is a revisit of a previous moment.

### Context for this conversation:
{context}

### Current conversation:
User: {request.query}

### Your Response Guidelines:
1. Focus specifically on this replay context
2. Acknowledge this is a revisit of a past moment
3. Connect the current query to the original reflection
4. Offer new perspective while honoring past insights
5. Keep response under 100 words
6. Speak in natural, caring language

Response:
""")
        
        # Generate response
        def generate_response():
            return llm.complete(prompt.format(user_name=user_name, context=context, query=request.query)).text
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = await loop.run_in_executor(executor, generate_response)
        
        return {"result": response.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Replay chat failed: {e}")
        return {"result": f"I had trouble accessing that memory. Let's try again?"}
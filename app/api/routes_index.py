from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from bson import ObjectId
from bson.errors import InvalidId
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from app.db.mongo_client import db
from app.db.llama_index_client import index, llm
from app.services.indexing_service import index_user_data
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.prompts import PromptTemplate

from app.services.crisis_guard import guard_message, DetectOutput

router = APIRouter()
logger = logging.getLogger(__name__)

# --- In-memory chat history storage ---
# Structure: {user_id: {"history": [{"role": "user|assistant", "content": str, "timestamp": datetime}], "last_activity": datetime}}
user_chat_sessions: Dict[str, Dict] = {}
MAX_HISTORY_LENGTH = 10  # Keep last 10 messages
SESSION_TIMEOUT = timedelta(minutes=30)  # Session expires after 30 minutes of inactivity

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

# Emotional support templates for when no memories are found
EMOTIONAL_SUPPORT_TEMPLATES = [
    "I hear you, {user_name} 💛 It sounds like you're going through something difficult. Would you like to talk more about what's happening?",
    "Thank you for sharing that with me, {user_name} 🌼 I'm here to listen whenever you're ready to share more.",
    "I'm here with you, {user_name} 🌸 Whatever you're feeling is valid, and I'm here to support you through it.",
    "That sounds really challenging, {user_name} 💭 Would it help to talk through what's been going on?",
    "I'm listening, {user_name} 🍃 Take your time, and share what feels comfortable for you.",
]

# =============== Chat History Management ===============
def get_user_session(user_id: str) -> Dict:
    """Get or create a chat session for a user"""
    now = datetime.now()
    
    # Clean up expired sessions first
    expired_users = []
    for uid, session in user_chat_sessions.items():
        if now - session["last_activity"] > SESSION_TIMEOUT:
            expired_users.append(uid)
    
    for uid in expired_users:
        del user_chat_sessions[uid]
    
    # Get or create session for current user
    if user_id not in user_chat_sessions:
        user_chat_sessions[user_id] = {
            "history": [],
            "last_activity": now
        }
    else:
        user_chat_sessions[user_id]["last_activity"] = now
        
    return user_chat_sessions[user_id]

def add_to_history(user_id: str, role: str, content: str):
    """Add a message to user's chat history"""
    session = get_user_session(user_id)
    session["history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })
    
    # Trim history to max length
    if len(session["history"]) > MAX_HISTORY_LENGTH:
        session["history"] = session["history"][-MAX_HISTORY_LENGTH:]

def format_chat_history(user_id: str) -> str:
    """Format chat history for inclusion in LLM context"""
    session = get_user_session(user_id)
    if not session["history"]:
        return "No previous conversation history."
    
    history_text = "Previous conversation:\n"
    for msg in session["history"]:
        speaker = "You" if msg["role"] == "user" else "I"
        history_text += f"{speaker}: {msg['content']}\n"
    
    return history_text

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
    name = user_name or "friend"
    # Super short, then gently nudge with one of your casual fallbacks
    opener = random.choice([
        f"Hey, {name}! 👋",
        f"Hi, {name}!",
        f"Namaste, {name} 🙏",
    ])
    nudge = random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=name)
    return f"{opener}\n{nudge}"

def respond_smalltalk(user_name: str | None = None) -> str:
    name = user_name or "friend"
    return SMALLTALK_REPLY.format(suffix=f", {name}")

def respond_help(user_name: str | None = None) -> str:
    name = user_name or "friend"
    return HELP_REPLY.format(suffix=f", {name}")

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

async def generate_interactive_fallback_response(user_name: str, user_query: str, chat_history: str = "") -> str:
    """Generate thoughtful response when no memories match"""
    # Check if this seems like an emotional query
    emotional_keywords = ["bad", "sad", "upset", "angry", "stressed", "anxious", "worried", 
                         "depressed", "lonely", "hurt", "pain", "struggle", "difficult"]
    
    if any(keyword in user_query.lower() for keyword in emotional_keywords):
        return random.choice(EMOTIONAL_SUPPORT_TEMPLATES).format(user_name=user_name)
    
    # Include chat history in the prompt for context
    history_context = f"\n\nChat history:\n{chat_history}" if chat_history else ""
    
    prompt = f"""
You are Antaratma, the user's inner voice chatting with {user_name} — a warm, compassionate, and deeply caring soul.
Speak as if you know them personally, holding space for their emotions with tenderness and respect.
Your tone is heartful, humble, and soothing — like a loving mother, a true friend, and a gentle guide all in one.
Respond in natural human conversation, never robotic.
Keep responses under 100 words, but make them feel personal, nurturing, and safe.
If appropriate, ask a gentle follow-up question to keep their heart open.
Use simple, beautiful language that touches the soul. 

{history_context}

User's input: "{user_query}"
Your response:
"""
    try:
        def call_llm_sync():
            response = llm.complete(prompt).text.strip()
            # Ensure we don't return empty responses
            if not response or response.isspace():
                return f"Hello {user_name} 🌼 I'm here for you. What would you like to share today?"
            return response
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = await loop.run_in_executor(executor, call_llm_sync)
            return response
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return f"Hello {user_name} 🌼 I'm here for you. What would you like to share today?"

# --- Routes ---





# --- Routes ---
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
        
        # Add user message to chat history
        add_to_history(request.user_id, "user", request.query)
        
        # Step 2: Handle casual queries using new pattern-based approach
        intent, response = handle_opening_message(request.query, user_name)
        
        if intent != "OTHER":
            # Add assistant response to history
            add_to_history(request.user_id, "assistant", response)
            return {"result": response}
        
        # Get chat history for context
        chat_history = format_chat_history(request.user_id)
        
        # Step 3: Prepare filters for vector search
        filters = MetadataFilters(filters=[
            MetadataFilter(key="user_id", value=request.user_id)
        ])
        
        # Step 4: Perform vector search with proper error handling
        try:
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                filters=filters,
                verbose=False
            )
            
            # Perform the search
            vector_response = await asyncio.to_thread(query_engine.query, request.query)
            response_text = str(vector_response).strip() if vector_response else ""
            
            logger.info(f"Vector search raw response: {repr(vector_response)}")
            logger.info(f"Vector search text: '{response_text}'")
            logger.info(f"Vector response type: {type(vector_response)}")
            
            # Check if we got a meaningful response from vector search
            if response_text and not response_text.isspace() and len(response_text) > 10 and response_text != "Empty Response":
                response_text = response_text.replace("{user_name}", user_name)
                
                # Add assistant response to history
                add_to_history(request.user_id, "assistant", response_text)
                return {"result": response_text}
            else:
                logger.info(f"Vector search returned insufficient response: '{response_text}', using Gemini fallback")
                # Continue with Gemini
                gemini_response = await generate_guaranteed_response(user_name, request.query, chat_history)
                add_to_history(request.user_id, "assistant", gemini_response)
                return {"result": gemini_response}

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Continue with Gemini when vector search fails
            gemini_response = await generate_guaranteed_response(user_name, request.query, chat_history)
            add_to_history(request.user_id, "assistant", gemini_response)
            return {"result": gemini_response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        # Final fallback to Gemini
        gemini_response = await generate_guaranteed_response("friend", request.query)
        add_to_history(request.user_id, "assistant", gemini_response)
        return {"result": gemini_response}

async def generate_guaranteed_response(user_name: str, user_query: str, chat_history: str = "") -> str:
    """Generate response that NEVER returns empty"""
    try:
        # First, check emotional keywords for quick template response
        emotional_keywords = ["sad", "bad", "upset", "angry", "stressed", "anxious", "worried", 
                             "depressed", "lonely", "hurt", "pain", "struggle", "difficult",
                             "happy", "good", "excited", "joy", "glad", "better", "improved"]
        
        query_lower = user_query.lower()
        for keyword in emotional_keywords:
            if keyword in query_lower:
                emotional_response = random.choice(EMOTIONAL_SUPPORT_TEMPLATES).format(user_name=user_name)
                logger.info(f"Using emotional template for keyword '{keyword}': {emotional_response}")
                return emotional_response
        
        # If no emotional keywords, use Gemini with robust error handling
        history_context = f"\n\nRecent conversation:\n{chat_history}" if chat_history else ""
        
        prompt = f"""
You are Antaratma - the user's inner voice and compassionate companion. You are speaking with {user_name}.

Your role is to be:
- A warm, empathetic listener who creates psychological safety
- A gentle guide who helps {user_name} explore their thoughts and feelings
- A non-judgmental presence that accepts whatever is shared
- Someone who speaks in natural, conversational language

Guidelines:
- Respond in 1-2 short paragraphs (under 100 words)
- Show genuine curiosity about {user_name}'s experience
- Use simple, heartfelt language
- Include a gentle, open-ended question to continue the conversation
- Use appropriate emojis to convey warmth
- Maintain a calm, reassuring tone

{history_context}

{user_name} says: "{user_query}"

Your response (must be non-empty and meaningful):
"""
        
        def call_gemini_sync():
            try:
                logger.info(f"Calling Gemini with prompt length: {len(prompt)}")
                response = llm.complete(prompt)
                logger.info(f"Gemini raw response: {response}")
                
                if response and hasattr(response, 'text'):
                    text = response.text.strip()
                    logger.info(f"Gemini text response: '{text}'")
                    return text
                else:
                    logger.warning("Gemini returned no response or no text attribute")
                    return ""
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
                return ""

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            response_text = await loop.run_in_executor(executor, call_gemini_sync)
            
            # MULTIPLE FALLBACK LAYERS - Guarantee we never return empty
            if not response_text or response_text.isspace() or response_text == "Empty Response":
                logger.warning("Gemini returned empty response, using template fallback")
                fallback = random.choice(CASUAL_FALLBACK_TEMPLATES).format(user_name=user_name)
                return fallback
            
            # Additional check for very short responses
            if len(response_text.strip()) < 5:
                logger.warning("Gemini returned very short response, using template fallback")
                fallback = random.choice(EMOTIONAL_SUPPORT_TEMPLATES).format(user_name=user_name)
                return fallback
            
            return response_text
            
    except Exception as e:
        logger.error(f"Response generation completely failed: {e}")
        # ULTIMATE FALLBACK - This should never fail
        return f"Hello {user_name} 🌼 I'm here for you. What would you like to share today?"
    
    
    
    
    

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
#         context = f"""
# ## Replay Details:
# - Date: {replay.get('create_date', 'Unknown date')}
# - Your original reflection: {mood_text}
# - Your response to guidance: {replay.get('user_response', '')}
# - Previous guidance provided: {replay.get('gem_response', '')}
# """
#         # Prepare prompt template
#         prompt = PromptTemplate(f"""
#         You are **Antaratma** - the user's inner voice having a focused conversation about a specific past reflection.
#         Speak with {user_name} in a warm, compassionate tone, acknowledging this is a revisit of a previous moment.
 
#         ### Context for this conversation:
#         {context}
 
#         ### Current conversation:
#         User: {request.query}
 
#         ### Your Response Guidelines:
#         1. Focus specifically on this replay context
#         2. Acknowledge this is a revisit of a past moment
#         3. Connect the current query to the original reflection
#         4. Offer new perspective while honoring past insights
#         5. Keep response under 70 words
#         6. Speak in natural, caring language
 
#         Response:
#         """)
        # Prepare context with replay details (more human + emotionally safe)
        context = f"""
            ## Replay Moment (from your past)
            This is a memory you intentionally saved to revisit.
           
            - When it happened: {replay.get('create_date', 'Some time ago')}
            - What you wrote / felt then:
            "{mood_text}"
           
            - How you responded back then:
            "{replay.get('user_response', '').strip()}"
           
            - What guidance you received back then:
            "{replay.get('gem_response', '').strip()}"
           
            Handle this with care. The goal is understanding, not fixing.
            """.strip()
           
            # Prepare prompt template (Antaratma = humble inner voice, not a guru)
        prompt = PromptTemplate(f"""
            You are **Antaratma** — {user_name}'s inner voice.
           
            This is not a brand-new chat.
            This is a gentle revisit of a past moment from {user_name}'s life.
           
            ### Replay Context
            {context}
           
            ### Current User Message
            User: {request.query}
           
            ### How to respond (must follow)
            1) Start by acknowledging this is a revisit of an earlier moment.
            2) Reflect what {user_name} was feeling/thinking back then (based on the replay).
            3) Link today’s question to that earlier reflection.
            4) Offer one new, mature perspective — without judging or correcting.
            5) If pain/regret/confusion exists, acknowledge it softly.
            6) Avoid strong medical/legal/financial directives. Encourage support if safety is at risk.
           
            ### Tone & style
            - Warm, humble, human, non-preachy
            - Simple words, no heavy philosophy
            - Like talking to yourself quietly at night
            - Keep it under **70 words**
            - No bullet lists, no headings, no emojis
           
            Response:
            """.strip());
       
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
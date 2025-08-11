from fastapi import APIRouter, HTTPException
from app.models.schemas import TextRequest
from app.services.emotion_service import analyze_emotion, detect_mood_and_events
from app.db.mongo_client import db
from bson import ObjectId
from fastapi import HTTPException
from fastapi import Body
from app.models.schemas import MoodCreateRequest
from app.services.indexing_service import index_user_data
from app.services.replay_service import build_replay  # assuming your replay logic is here




router = APIRouter()

@router.post("/analyze")
async def analyze(req: TextRequest):
    return analyze_emotion(req.text)

@router.post("/detect-mood/text")
def detect_mood_route(req: TextRequest):
    return detect_mood_and_events(req.text)

def serialize_mongo_doc(doc):
    doc["_id"] = str(doc["_id"])
    if "user" in doc and isinstance(doc["user"], ObjectId):
        doc["user"] = str(doc["user"])
    return doc

# Helper function to serialize MongoDB ObjectIds into strings
def serialize_mongo_rep(doc):
    if isinstance(doc, list):
        return [serialize_mongo_rep(item) for item in doc]
    elif isinstance(doc, dict):
        return {k: serialize_mongo_rep(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc


@router.get("/moods")
async def get_user_moods(user_id: str):
    try:
        object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    moods_cursor = db.moods.find({"user": object_id}).sort("create_date", -1)
    moods = await moods_cursor.to_list(100)

    serialized = [serialize_mongo_doc(m) for m in moods]

    print(f"[INFO] Retrieved {len(serialized)} moods for user_id: {user_id}")
    return serialized



@router.get("/collections")
async def list_collections():
    collections = await db.list_collection_names()
    print(f"[INFO] Available collections: {collections}")
    return {"collections": collections}


@router.post("/mood-detect")
async def create_mood_with_replay(mood_data: MoodCreateRequest):
    try:
        user_object_id = ObjectId(mood_data.user)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    # Insert mood
    mood_dict = mood_data.dict()
    mood_dict["user"] = user_object_id
    mood_result = await db.moods.insert_one(mood_dict)
    created_mood = await db.moods.find_one({"_id": mood_result.inserted_id})

    # Build replay using same logic as /replay
    context = {
        "mood_today": created_mood.get("mood"),
        "user_location": {
            "lat": created_mood.get("latitude"),
            "lng": created_mood.get("longitude"),
        },
        "today_date": created_mood.get("create_date"),
    }
    replay_generated = build_replay(created_mood, context)  # returns ai_response, context_tags, location, replay_opportunity_score

    replay_payload = {
        "user_response": created_mood.get("user_text", ""),
        "mood": created_mood.get("mood"),
        "gem_response": replay_generated.get("ai_response"),
        "user": user_object_id,
        "is_shown": created_mood.get("is_shown"),
        "longitude": created_mood.get("longitude"),
        "latitude": created_mood.get("latitude"),
        "events": created_mood.get("events", []),
        "context_tags": replay_generated.get("context_tags", []),
        "replay_opportunity_score": str(replay_generated.get("replay_opportunity_score", "0")),
        "moods": created_mood["_id"],
        "location": replay_generated.get("location"),
        "create_date": created_mood.get("create_date"),
    }

    replay_result = await db.replays.insert_one(replay_payload)
    created_replay = await db.replays.find_one({"_id": replay_result.inserted_id})

    try:
        await index_user_data(
            user_id=str(user_object_id),
            moods=[created_mood],
            replays=[created_replay]
        )
    except Exception as e:
        print(f"❌ Indexing failed: {e}")

    return {
        "mood": serialize_mongo_doc(created_mood),
        "replay": serialize_mongo_rep(created_replay)
    }




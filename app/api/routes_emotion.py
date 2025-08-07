from fastapi import APIRouter, HTTPException
from app.models.schemas import TextRequest
from app.services.emotion_service import analyze_emotion, detect_mood_and_events
from app.db.mongo_client import db
from bson import ObjectId
from fastapi import HTTPException
from fastapi import Body
from app.models.schemas import MoodCreateRequest
from app.services.indexing_service import index_user_data




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

@router.post("/moods")
async def create_mood(mood_data: MoodCreateRequest):
    try:
        user_object_id = ObjectId(mood_data.user)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    # Prepare mood dict for MongoDB
    mood_dict = mood_data.dict()
    mood_dict["user"] = user_object_id

    # Save to MongoDB
    result = await db.moods.insert_one(mood_dict)
    created_mood = await db.moods.find_one({"_id": result.inserted_id})

    # Index this new mood
    try:
        await index_user_data(
            user_id=str(user_object_id),
            moods=[created_mood],  # Index only this mood
            replays=[]             # No replays in this case
        )
        print(f"✅ Indexed new mood for user {user_object_id}")
    except Exception as e:
        print(f"❌ Failed to index new mood: {e}")

    return serialize_mongo_doc(created_mood)

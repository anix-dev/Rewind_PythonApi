from fastapi import APIRouter, HTTPException
from bson import ObjectId

from app.models.schemas import ReplayRequest, ReplayCreateRequest
from app.services.replay_service import build_replay
from app.db.mongo_client import db

router = APIRouter()


# Helper function to serialize MongoDB ObjectIds into strings
def serialize_mongo_doc(doc):
    if isinstance(doc, list):
        return [serialize_mongo_doc(item) for item in doc]
    elif isinstance(doc, dict):
        return {k: serialize_mongo_doc(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc


@router.post("/replay")
def generate_replay(request: ReplayRequest):
    """
    Generates an emotionally reflective replay message using the user input.
    This does not store data in MongoDB.
    """
    sample = request.dict()

    context = {
        "mood_today": sample["mood"],
        "user_location": {
            "lat": sample["latitude"],
            "lng": sample["longitude"]
        },
        "today_date": sample["create_date"]
    }

    replay = build_replay(sample, context)
    return replay


@router.get("/user-replay")
async def get_user_replays(user_id: str):
    """
    Retrieves all replay records for a given user from the MongoDB collection.
    """
    try:
        object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    replays = await db.replays.find({"user": object_id}).sort("create_date", -1).to_list(100)
    serialized = serialize_mongo_doc(replays)

    print(f"[INFO] Retrieved {len(serialized)} replays for user_id: {user_id}")
    return serialized


@router.post("/user-replay")
async def create_user_replay(replay_data: ReplayCreateRequest):
    """
    Stores a user-generated replay in the MongoDB database.
    Requires valid ObjectIds for user and moods fields.
    """
    try:
        user_object_id = ObjectId(replay_data.user)
        moods_object_id = ObjectId(replay_data.moods)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId for user or moods")

    replay_dict = replay_data.dict()
    replay_dict["user"] = user_object_id
    replay_dict["moods"] = moods_object_id

    result = await db.replays.insert_one(replay_dict)
    created_replay = await db.replays.find_one({"_id": result.inserted_id})

    return serialize_mongo_doc(created_replay)

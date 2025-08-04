from fastapi import APIRouter
from app.models.schemas import ReplayRequest
from app.services.replay_service import build_replay
from app.db.mongo_client import db



from bson import ObjectId
from fastapi import APIRouter, HTTPException

router = APIRouter()

def serialize_mongo_doc(doc):
    if isinstance(doc, list):
        return [serialize_mongo_doc(item) for item in doc]
    elif isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            new_doc[k] = serialize_mongo_doc(v)
        return new_doc
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc




@router.post("/replay")
def generate_replay(request: ReplayRequest):
    """
    Generates an emotionally reflective replay message using the user input.
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
    try:
        object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    replays = await db.replays.find({"user": object_id}).sort("create_date", -1).to_list(100)
    serialized = serialize_mongo_doc(replays)

    print(f"[INFO] Retrieved {len(serialized)} replays for user_id: {user_id}")
    return serialized


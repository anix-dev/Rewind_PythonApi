from fastapi import APIRouter
from app.models.schemas import ReplayRequest
from app.services.replay_service import build_replay

router = APIRouter()


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

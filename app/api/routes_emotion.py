from fastapi import APIRouter, HTTPException
from app.models.schemas import TextRequest
from app.services.emotion_service import analyze_emotion, detect_mood_and_events

router = APIRouter()

@router.post("/analyze")
async def analyze(req: TextRequest):
    return analyze_emotion(req.text)

@router.post("/detect-mood/text")
def detect_mood_route(req: TextRequest):
    return detect_mood_and_events(req.text)

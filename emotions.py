# emotions.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import json

# Load taxonomy
with open("emotions.json", "r", encoding="utf-8") as f:
    emotion_json = json.load(f)

label_map = {
    "joy": "Happy",
    "love": "Happy",      
    "sadness": "Sad",
    "anger": "Angry",
    "fear": "Fearful",
    "surprise": "Surprised",
    "disgust": "Disgust",
    "neutral": "Neutral" 
}

emotion_pipe = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

def traverse_and_find(label, node):
    if node.get("feeling", "").lower() == label.lower():
        return node
    for key in ["secondary", "tertiary"]:
        if key in node:
            for child in node[key]:
                found = traverse_and_find(label, child)
                if found:
                    return found
    return None

def get_emotion_details(emotion_label, emotions_json):
    for root in emotions_json:
        found = traverse_and_find(emotion_label, root)
        if found:
            return found
    return None

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/analyze")
async def analyze_emotion(req: TextRequest):
    results = emotion_pipe(req.text)[0]
    pred = max(results, key=lambda x: x['score'])
    base_label = pred['label'].lower()
    mapped_label = label_map.get(base_label, base_label)
    emotion_data = get_emotion_details(mapped_label, emotion_json)
    if not emotion_data:
        raise HTTPException(status_code=404, detail=f"Emotion '{mapped_label}' not found in taxonomy.")
    return {
        "input_text": req.text,
        "taxonomy_label": mapped_label,
        "taxonomy_emotion": emotion_data.get("feeling"),
        "score": float(pred['score']),
        "description": emotion_data.get("description"),
        "example": emotion_data.get("example_scenario"),
        "synonyms": emotion_data.get("synonyms"),
        "antonyms": emotion_data.get("antonyms"),
        "intensity": emotion_data.get("intensity"),
        "category": emotion_data.get("category"),
        "physical_sensations": emotion_data.get("physical_sensations", []),
    }

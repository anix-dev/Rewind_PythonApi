# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load emotion detection model
emotion_pipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# Request model
class TextRequest(BaseModel):
    text: str

@app.post("/detect-mood/text")
def detect_mood(data: TextRequest):
    results = emotion_pipeline(data.text)
    top_result = results[0]
    return {
        "mood": top_result["label"],
        "confidence": round(top_result["score"], 4)
    }

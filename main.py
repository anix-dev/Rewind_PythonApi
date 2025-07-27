from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import spacy
from typing import List
from datetime import datetime
import random
from replay_engine import build_replay
from datetime import datetime
from pydantic import BaseModel
from typing import List
from emotions import router as emotions_router
app = FastAPI()

# Load models
emotion_pipeline = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
nlp = spacy.load("en_core_web_sm")


app.include_router(emotions_router)

class ReplayRequest(BaseModel):
    user_text: str
    mood: str
    longitude: float
    latitude: float
    events: List[str]
    context_tags: List[str]
    create_date: str


@app.post("/replay")
def generate_replay(request: ReplayRequest):
    sample = request.dict()
    context = {
        "mood_today": sample["mood"],
        "user_location": {
            "lat": sample["latitude"],
            "lng": sample["longitude"]
        },
        "today_date": datetime.utcnow().strftime('%Y-%m-%d')
    }
    replay = build_replay(sample, context)
    return replay

class TextRequest(BaseModel):
    text: str

# Keyword groups
keyword_categories = {
    "life_goal": [
        "dream", "goal", "aspire", "ambition", "bucket list", "always wanted", "before I die"
    ],
    "travel_event": [
        "trip", "travel", "vacation", "journey", "tour", "destination", "flight", "hotel",
        "goa", "europe", "mountains", "beach", "desert safari", "hill station", "paris", "london", "trek", "camping"
    ],
    "missed_event": [
        "missed", "forgot", "couldn't", "didn't", "left out", "cancelled", "skipped", "was late for"
    ],
    "special_day": [
        "birthday", "anniversary", "special day", "valentine", "festival", "new year", "eid", "diwali", "christmas", "holi"
    ],
    "milestone": [
        "graduation", "promotion", "wedding", "achievement", "won", "completed", "retired",
        "first job", "milestone", "passed exam", "trophy", "medal", "award", "certification"
    ],
    "relationship_event": [
        "friend", "partner", "boyfriend", "girlfriend", "wife", "husband", "son", "daughter", "mom", "dad", "parents",
        "family", "sibling", "teacher", "colleague", "boss"
    ],
    "reflection": [
        "remember", "recall", "think back", "reflected", "looking back", "used to", "reminiscing", "memory", "nostalgia"
    ],
    "loss_or_challenge": [
        "lost", "failure", "death", "passed away", "struggled", "broke", "hurt", "sick", "injury",
        "pain", "accident", "disappointed", "rejected", "heartbreak", "illness", "depression","breakup"
    ],
    "sports_event": [
        "match", "game", "tournament", "won", "lost", "scored", "cricket", "football", "badminton",
        "played", "team", "goal", "innings", "batting", "bowling", "umpire", "stadium"
    ],
     "health": [
        "health", "fitness", "workout", "gym", "exercise", "yoga", "diet", "meditation", "wellness",
        "doctor", "hospital", "surgery", "therapy", "recovery", "illness", "medicine", "pain", "checkup", "mental health"
    ],
      "health": [
        "exercise", "gym", "fitness", "diet", "yoga", "workout", "wellness", "checkup", "health issue", "doctor",
        "hospital", "therapy", "mental health", "recovery", "meditation", "medicine", "clinic", "surgery", "nutrition"
    ]
}


context_keywords = {
    "people": [
        "friend", "mom", "dad", "wife", "husband", "brother", "sister", "child", "teacher",
        "daughter", "son", "partner", "boss", "colleague", "classmate", "cousin", "mentor"
    ],
    "occasions": [
        "birthday", "anniversary", "wedding", "graduation", "festival", "valentine",
        "new year", "christmas", "diwali", "eid", "holi", "raksha bandhan"
    ],
    "travel": [
        "trip", "vacation", "holiday", "traveled", "missed flight", "journey", "goa",
        "paris", "london", "hill station", "trek", "camping", "desert safari", "resort"
    ],
    "regret": [
        "forgot", "missed", "should have", "couldn’t", "didn't", "could not", "left out", "lost chance",
        "regret", "skipped", "was late", "canceled", "rejected"
    ],
    "career": [
        "job", "interview", "promotion", "resigned", "project", "boss", "client", "deadline",
        "salary", "workplace", "colleague", "fired", "hired"
    ],
    "education": [
        "exam", "test", "result", "school", "college", "teacher", "grade", "admission", "assignment",
        "passed", "failed", "semester", "syllabus", "tuition"
    ],
    "emotion": [
        "happy", "sad", "angry", "excited", "lonely", "depressed", "anxious", "proud", "ashamed",
        "joyful", "scared", "relieved", "worried", "regretful"
    ],
    "dreams": [
        "dreamed of", "bucket list", "wanted to", "goal", "ambition", "hope", "aspire", "vision",
        "desire", "plan", "wish"
    ],
    "dates": [
        "birthday", "anniversary", "special day", "important day", "today", "yesterday", "last year", "this year"
    ],
    "sports": [
        "cricket", "football", "match", "tournament", "game", "score", "team", "win", "won", "lost",
        "goal", "stadium", "bat", "ball", "umpire", "coach", "play", "innings", "kick", "field"
    ]
}


# Helper Functions
def detect_event_categories(text: str):
    tags = []
    lowered = text.lower()
    for category, keywords in keyword_categories.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(category)
    return list(set(tags))

def extract_time(text: str):
    lowered = text.lower()
    if "last year" in lowered: return "last year"
    if "college" in lowered: return "college"
    if "school" in lowered: return "school"
    if "yesterday" in lowered: return "yesterday"
    if "today" in lowered: return "today"
    if "birthday" in lowered: return "birthday"
    return "unknown"

def extract_life_events(text):
    doc = nlp(text)
    events = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        categories = detect_event_categories(sentence_text)
        if categories:
            for category in categories:
                events.append({
                    "event": category.replace("_", " ").title(),
                    "title": sentence_text,
                    "time": extract_time(sentence_text),
                    "status": "mentioned",
                    "category": category
                })
    return events

def extract_context_tags(text: str) -> List[str]:
    tags = set()
    lowered = text.lower()
    for keyword_list in context_keywords.values():
        for keyword in keyword_list:
            if keyword in lowered:
                tags.add(keyword)
    return list(tags)

def generate_summary(text: str) -> str:
    text = text.lower()
    if "birthday" in text and ("forgot" in text or "missed" in text):
        return "Missed a birthday and feeling regretful."
    elif "trip" in text or "vacation" in text:
        if "missed" in text or "cancelled" in text:
            return "Missed a travel plan."
        return "Recollecting a travel experience."
    elif "exam" in text or "result" in text:
        return "Reflecting on an academic milestone."
    elif "job" in text or "promotion" in text:
        return "Career reflection."
    elif "forgot" in text or "missed" in text:
        return "Missed something important."
    elif "goal" in text or "dream" in text:
        return "Thinking about personal dreams or goals."
    elif "happy" in text or "excited" in text:
        return "A joyful moment."
    elif "sad" in text or "regret" in text:
        return "A moment of sadness or regret."
    return "Reflecting on a personal memory."

def generate_replay_opportunity_score(memory: dict) -> float:
    score = 0.4
    mood = memory.get("mood", "").lower()
    if mood in ["sadness", "regret"]: score += 0.2
    elif mood in ["nostalgic", "reflective"]: score += 0.1

    events = memory.get("events", [])
    if "missed_event" in events: score += 0.2
    if "special_day" in events: score += 0.1

    tags = memory.get("context_tags", [])
    relationship_tags = {"friend", "birthday", "daughter", "son", "wife", "partner", "mom", "dad"}
    if any(tag in tags for tag in relationship_tags): score += 0.1

    return round(min(score, 1.0), 2)

# Final Endpoint
@app.post("/detect-mood/text")
def detect_mood(data: TextRequest):
    text = data.text
    emotion_result = emotion_pipeline(text)[0]
    mood = emotion_result["label"]
    confidence = round(emotion_result["score"], 4)

    events = extract_life_events(text)
    event_types = list({e["category"] for e in events})

    context_tags = extract_context_tags(text)
    summary = generate_summary(text)

    memory_data = {
        "user_text": text,
        "mood": mood,
        "events": event_types,
        "context_tags": context_tags
    }
    replay_opportunity_score = generate_replay_opportunity_score(memory_data)

    return {
        "emotion": mood,
        "confidence": confidence,
        "summary": summary,
        "context_tags": context_tags,
        "replay_opportunity_score": replay_opportunity_score,
        "detectedEvents": events
    }

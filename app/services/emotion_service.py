from transformers import pipeline
import spacy
from typing import List, Dict
import re

# RapidFuzz for typo-tolerant keyword matching
from rapidfuzz import fuzz


# ---------------- LOAD MODELS ---------------- #

# Emotion model
emotion_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,              # modern replacement for return_all_scores=True
    truncation=True
)

# Sentiment model to avoid false "anger" on neutral/basic inputs
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True
)

nlp = spacy.load("en_core_web_sm")


# ---------------- TEXT NORMALIZATION ---------------- #

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)               # collapse spaces
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)     # sooo -> soo (soften repeats)
    return text


def fuzzy_contains(text: str, phrase: str, threshold: int = 90) -> bool:
    """
    Returns True if phrase is present or close enough (typo tolerant).
    Good for: "wissh i  had" ~ "wish i had"
    """
    text_n = normalize_text(text)
    phrase_n = normalize_text(phrase)

    if not phrase_n:
        return False

    if phrase_n in text_n:
        return True

    # partial_ratio is good for substring-ish matching
    return fuzz.partial_ratio(phrase_n, text_n) >= threshold


def _has_any(text: str, phrases: List[str], threshold: int = 90) -> bool:
    return any(fuzzy_contains(text, p, threshold=threshold) for p in phrases)


def _anger_like_formatting(text: str) -> bool:
    if not text:
        return False
    # ALL CAPS or many exclamation marks can indicate intensity
    if len(text) >= 8 and text.upper() == text and any(c.isalpha() for c in text):
        return True
    if text.count("!") >= 2:
        return True
    return False


# ---------------- KEYWORDS (EXPANDED) ---------------- #

keyword_categories = {
    "life_goal": [
        "dream", "dreams", "goal", "goals", "aspire", "aspiring", "ambition", "ambitions",
        "bucket list", "always wanted", "long wanted", "lifelong dream", "before i die",
        "one day i will", "someday i will", "want to become", "i want to be",
        "my purpose", "my calling", "mission in life"
    ],

    "travel_event": [
        "trip", "travel", "travelling", "vacation", "holiday", "journey", "tour", "getaway",
        "destination", "itinerary", "flight", "boarding pass", "hotel", "resort", "airbnb",
        "road trip", "train", "visa", "passport", "check-in", "luggage",
        "mountains", "beach", "desert safari", "hill station", "trek", "trekking", "hike", "camping",
        "goa", "manali", "shimla", "kasol", "leh", "ladakh", "paris", "london", "europe"
    ],

    "missed_event": [
        "missed", "forgot", "couldn't", "could not", "didn't", "did not",
        "left out", "cancelled", "canceled", "skipped", "was late", "late for",
        "could not make it", "couldn't make it", "missed out", "i wish i went", "should have gone"
    ],

    "special_day": [
        "birthday", "bday", "anniversary", "special day", "valentine", "valentine's day",
        "festival", "new year", "new year's", "eid", "diwali", "christmas", "holi",
        "rakhi", "raksha bandhan", "karwa chauth", "navratri",
        "first date", "date night", "engagement day", "proposal day",
        "baby shower", "housewarming", "farewell", "reunion"
    ],

    "milestone": [
        "graduation", "graduated", "promotion", "promoted", "wedding", "marriage",
        "achievement", "achieved", "won", "completed", "retired",
        "first job", "new job", "job offer", "milestone", "passed exam", "cleared exam",
        "trophy", "medal", "award", "certification", "certificate",
        "started a business", "launched", "got selected", "rank", "result day"
    ],

    "relationship_event": [
        "friend", "friends", "partner", "boyfriend", "girlfriend", "wife", "husband",
        "son", "daughter", "mom", "mother", "dad", "father", "parents",
        "family", "sibling", "brother", "sister", "teacher", "mentor",
        "colleague", "boss", "manager", "team", "crush", "ex", "fiancé", "fiance", "spouse"
    ],

    "reflection": [
        "remember", "recall", "think back", "reflected", "reflection",
        "looking back", "back then", "used to", "reminiscing", "memory", "nostalgia",
        "i still think about", "it reminds me", "flashback", "throwback"
    ],

    "loss_or_challenge": [
        "lost", "loss", "failure", "failed", "death", "passed away", "gone forever",
        "struggled", "hard time", "broke", "hurt", "injury", "pain",
        "accident", "disappointed", "rejected", "heartbreak", "broken",
        "illness", "sick", "depression", "anxiety", "panic",
        "breakup", "separation", "divorce", "lonely", "grief"
    ],

    "sports_event": [
        "match", "game", "tournament", "final", "semi final", "practice",
        "won", "lost", "scored", "team", "league",
        "cricket", "football", "badminton", "tennis", "basketball"
    ],

    "health": [
        "exercise", "workout", "gym", "fitness", "diet", "yoga", "meditation",
        "wellness", "checkup", "doctor", "hospital", "therapy", "counselling", "counseling",
        "mental health", "recovery", "medicine", "surgery",
        "sleep", "insomnia", "stress", "burnout", "headache", "fever"
    ]
}

context_keywords = {
    "people": [
        "friend", "friends", "mom", "mother", "dad", "father", "wife", "husband",
        "brother", "sister", "child", "kid", "daughter", "son", "partner",
        "boss", "manager", "colleague", "mentor", "teacher", "family", "parents"
    ],

    "regret": [
        "forgot", "missed", "should have", "could have", "regret", "skipped", "was late",
        "i wish", "if only", "wish i had", "wish i did", "wish i hadn't",
        "i messed up", "my mistake", "i shouldn't have", "shouldn't have", "could've"
    ],

    "career": [
        "job", "work", "promotion", "resigned", "quit", "project", "salary", "workplace",
        "boss", "manager", "office", "deadline", "interview", "offer", "client", "startup"
    ],

    "education": [
        "exam", "tests", "school", "college", "university", "passed", "failed",
        "result", "marks", "grades", "assignment", "homework", "semester"
    ],

    "sports": [
        "cricket", "football", "match", "tournament", "won", "lost", "score", "league"
    ]
}


# ---------------- MOOD RELIABILITY HELPERS ---------------- #

ANGER_CUES = [
    "angry", "furious", "rage", "mad", "pissed", "annoyed", "irritated",
    "hate", "disgusting", "stupid", "idiot", "shut up", "screw", "damn",
    "unacceptable", "fed up", "can't stand", "worst", "pathetic",
    "i'm done", "enough", "this is bullshit", "this is ridiculous", "so sick of"
]

FRUSTRATION_CUES = [
    "frustrated", "fed up", "tired of", "annoyed", "irritated",
    "stuck", "exhausted", "done with", "overwhelmed", "drained",
    "burnt out", "burned out", "can't deal", "not working", "keeps failing"
]

REGRET_CUES = [
    "should have", "could have", "wish i had", "regret", "missed chance",
    "i wish", "if only", "i shouldn't have", "i messed up", "my mistake",
    "wish i could", "wish i did", "wish i hadn't"
]

NEUTRAL_INTENT_CUES = [
    "how to", "what is", "can you", "please", "help", "explain", "steps",
    "guide", "suggest", "recommend", "tell me", "difference",
    "need info", "i want to understand", "how does it work"
]


# ---------------- CORE FUNCTIONS ---------------- #

def analyze_emotion(text: str) -> Dict:
    text_n = normalize_text(text)
    scores = emotion_pipeline(text_n)[0]  # list of dicts
    top = max(scores, key=lambda x: x["score"])
    ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
    return {
        "label": top["label"],
        "score": round(float(top["score"]), 4),
        "ranked": [{"label": s["label"], "score": round(float(s["score"]), 4)} for s in ranked]
    }


def analyze_sentiment(text: str) -> Dict:
    text_n = normalize_text(text)
    out = sentiment_pipeline(text_n)[0]  # POSITIVE/NEGATIVE
    return {"label": out["label"], "score": round(float(out["score"]), 4)}


def normalize_mood(text: str, emotion_label: str, emotion_score: float, sentiment: Dict) -> Dict:
    """
    Returns a pack:
      { "mood": str, "mood_confidence": float, "notes": str }
    """
    t = normalize_text(text)

    # If user is just asking neutral question, don't force negative emotions
    if _has_any(t, NEUTRAL_INTENT_CUES, threshold=92) and emotion_score < 0.60:
        return {"mood": "neutral", "mood_confidence": 0.65, "notes": "neutral_intent"}

    # Low confidence => neutral
    if emotion_score < 0.45:
        return {"mood": "neutral", "mood_confidence": emotion_score, "notes": "low_emotion_confidence"}

    # Cue-based overrides (typo tolerant)
    if _has_any(t, REGRET_CUES, threshold=88):
        return {"mood": "regret", "mood_confidence": max(emotion_score, 0.70), "notes": "regret_cue"}
    if _has_any(t, FRUSTRATION_CUES, threshold=88):
        return {"mood": "frustration", "mood_confidence": max(emotion_score, 0.70), "notes": "frustration_cue"}

    # Anger gating
    if emotion_label == "anger":
        sentiment_is_negative = (sentiment["label"] == "NEGATIVE" and sentiment["score"] >= 0.60)
        has_anger_cues = _has_any(t, ANGER_CUES, threshold=88) or _anger_like_formatting(text)

        # if it doesn't look like anger and sentiment isn't negative -> neutral
        if not has_anger_cues and not sentiment_is_negative:
            return {"mood": "neutral", "mood_confidence": 0.60, "notes": "anger_rejected"}

        # prefer "frustration" over "anger" in REWIND
        return {"mood": "frustration", "mood_confidence": emotion_score, "notes": "anger_to_frustration"}

    # Map model emotions to REWIND-friendly labels
    label_map = {
        "joy": "joy",
        "love": "love",
        "sadness": "sadness",
        "fear": "anxiety",
        "surprise": "surprise"
    }
    mapped = label_map.get(emotion_label, emotion_label)

    # Strong positive sentiment can soften borderline sadness/fear
    if sentiment["label"] == "POSITIVE" and sentiment["score"] >= 0.75 and emotion_score < 0.55:
        return {"mood": "neutral", "mood_confidence": 0.60, "notes": "positive_soften"}

    return {"mood": mapped, "mood_confidence": emotion_score, "notes": "emotion_mapped"}


def detect_event_categories(text: str) -> List[str]:
    t = normalize_text(text)
    found = set()
    for cat, keys in keyword_categories.items():
        for k in keys:
            if fuzzy_contains(t, k, threshold=90):
                found.add(cat)
                break
    return list(found)


def extract_time(text: str) -> str:
    t = normalize_text(text)
    time_terms = [
        "last year", "this year", "yesterday", "today", "tomorrow",
        "college", "school", "birthday", "anniversary"
    ]
    for term in time_terms:
        if term in t:
            return term
    return "unknown"


def extract_life_events(text: str) -> List[Dict]:
    doc = nlp(text or "")
    events = []
    for sent in doc.sents:
        categories = detect_event_categories(sent.text)
        for cat in categories:
            events.append({
                "event": cat.replace("_", " ").title(),
                "title": sent.text.strip(),
                "time": extract_time(sent.text),
                "status": "mentioned",
                "category": cat
            })
    return events


def extract_context_tags(text: str) -> List[str]:
    t = normalize_text(text)
    tags = set()
    for words in context_keywords.values():
        for w in words:
            if fuzzy_contains(t, w, threshold=90):
                tags.add(w)
    return list(tags)


def generate_summary(text: str) -> str:
    t = normalize_text(text)
    if "birthday" in t and ("missed" in t or "forgot" in t):
        return "Missed a birthday and feeling regretful."
    if "trip" in t or "vacation" in t or "travel" in t:
        return "Thinking about a travel experience."
    if "exam" in t or "result" in t:
        return "Reflecting on an academic experience."
    if "job" in t or "promotion" in t or "work" in t:
        return "Career-related reflection."
    if "first date" in t:
        return "Thinking about a special relationship moment."
    return "Reflecting on a personal memory."


def generate_replay_opportunity_score(memory: dict) -> float:
    score = 0.4
    mood = memory.get("mood", "")
    if mood in ["regret", "frustration", "sadness", "anxiety"]:
        score += 0.2
    if "missed_event" in memory.get("events", []):
        score += 0.2
    if "birthday" in memory.get("context_tags", []):
        score += 0.1
    return round(min(score, 1.0), 2)


def detect_mood_and_events(text: str) -> Dict:
    text = text or ""

    emotion_result = analyze_emotion(text)
    sentiment_result = analyze_sentiment(text)

    mood_pack = normalize_mood(
        text=text,
        emotion_label=emotion_result["label"],
        emotion_score=emotion_result["score"],
        sentiment=sentiment_result
    )

    events = extract_life_events(text)
    event_types = list({e["category"] for e in events})
    context_tags = extract_context_tags(text)
    summary = generate_summary(text)

    memory_data = {
        "mood": mood_pack["mood"],
        "events": event_types,
        "context_tags": context_tags
    }

    return {
        "emotion": mood_pack["mood"],
        "confidence": mood_pack["mood_confidence"],
        "summary": summary,
        "context_tags": context_tags,
        "replay_opportunity_score": generate_replay_opportunity_score(memory_data),
        "detectedEvents": events,

        # Keep this in staging to tune thresholds; remove in prod if you want
        "debug": {
            "raw_emotion_label": emotion_result["label"],
            "raw_emotion_confidence": emotion_result["score"],
            "sentiment": sentiment_result,
            "normalization_notes": mood_pack["notes"],
            "emotion_top3": emotion_result["ranked"][:3]
        }
    }
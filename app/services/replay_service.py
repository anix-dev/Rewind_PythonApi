import google.generativeai as genai
from datetime import datetime
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Tag categorization using keywords
keyword_categories = {
    "life_goal": ["dream", "goal", "aspire", "ambition", "bucket list", "always wanted", "before I die"],
    "travel_event": ["trip", "travel", "vacation", "journey", "tour", "destination", "flight", "hotel", "goa", "europe", "mountains", "beach", "desert safari", "hill station", "paris", "london", "trek", "camping"],
    "missed_event": ["missed", "forgot", "couldn't", "didn't", "left out", "cancelled", "skipped", "was late for"],
    "special_day": ["birthday", "anniversary", "special day", "valentine", "festival", "new year", "eid", "diwali", "christmas", "holi"],
    "milestone": ["graduation", "promotion", "wedding", "achievement", "won", "completed", "retired", "first job", "milestone", "passed exam", "trophy", "medal", "award", "certification"],
    "relationship_event": ["friend", "partner", "boyfriend", "girlfriend", "wife", "husband", "son", "daughter", "mom", "dad", "parents", "family", "sibling", "teacher", "colleague", "boss"],
    "reflection": ["remember", "recall", "think back", "reflected", "looking back", "used to", "reminiscing", "memory", "nostalgia"],
    "loss_or_challenge": ["lost", "failure", "death", "passed away", "struggled", "broke", "hurt", "sick", "injury", "pain", "accident", "disappointed", "rejected", "heartbreak", "illness", "depression"],
    "sports_event": ["match", "game", "tournament", "won", "lost", "scored", "cricket", "football", "badminton", "played", "team", "goal", "innings", "batting", "bowling", "umpire", "stadium"]
}


def get_location_name(latitude: float, longitude: float) -> str:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "zoom": 10
            },
            headers={"User-Agent": "mood-reflection-app"}
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("display_name", "Unknown location")
    except Exception:
        pass
    return "Unknown location"


def extract_tags(text: str):
    tags = []
    for category, keywords in keyword_categories.items():
        if any(word.lower() in text.lower() for word in keywords):
            tags.append(category)
    return tags


def score_replay_opportunity(text: str, mood: str) -> float:
    base_score = 0.4
    if "celebrate" in text or "won" in text:
        base_score += 0.3
    if mood.lower() in ["joy", "pride"]:
        base_score += 0.2
    elif mood.lower() in ["sadness", "regret"]:
        base_score += 0.1
    return min(base_score, 1.0)


def build_replay(data: dict, context: dict) -> dict:
    user_text = data.get("user_text", "")
    mood = data.get("mood", "")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    create_date = data.get("create_date", datetime.now().strftime("%Y-%m-%d"))

    # Extract context tags and score
    context_tags = extract_tags(user_text)
    replay_opportunity_score = score_replay_opportunity(user_text, mood)

    # Get location name
    location_name = get_location_name(latitude, longitude) if latitude and longitude else "Unknown location"

    # Prompt to Gemini model
    prompt = (
        f"You are an emotional reflection assistant for a journaling and memory replay app called REWIND.\n"
        f"Your goal is to generate a warm, emotionally intelligent `replay_message` (1–2 sentences) that encourages the user to reflect on and emotionally reconnect with a specific past memory.\n\n"
        f"Use the following inputs:\n"
        f"- User memory: '{user_text}'\n"
        f"- Mood: {mood}\n"
        f"- Location: {location_name}\n"
        f"- Context tags: {context_tags}\n"
        f"- Date of event: {create_date}\n\n"
        f"Instructions:\n"
        f"1. Acknowledge the significance of the moment emotionally (based on mood).\n"
        f"2. Mention the location or date if it adds personal or nostalgic weight.\n"
        f"3. Encourage the user to pause, reflect, or emotionally rewind that moment.\n"
        f"4. Keep it personal, thoughtful, and written in a warm and slightly poetic AI tone.\n"
        f"5. Do not repeat the exact user text. Rephrase meaningfully.\n\n"
        f"Now generate the `replay_message`."
    )

    try:
        response = model.generate_content(prompt)
        ai_response = response.text.strip() if response else "Here's a reflection opportunity for you."
    except Exception as e:
        ai_response = "Failed to generate reflection due to an internal error."

    return {
        "ai_response": ai_response,
        "replay_opportunity_score": replay_opportunity_score,
        "context_tags": context_tags,
        "location": location_name
    }

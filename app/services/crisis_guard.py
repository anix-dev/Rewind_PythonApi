
"""
Antaratma Crisis & Prohibited Content Guard (English-only)
- Category → keywords/regex
- Detector (EN/Hinglish/Hindi + Devanagari)
- Country-aware helpline resolver (remote-config first, fallback JSON)
- Safe, short responses per category (English)
- Cooldown to avoid spamming the user
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
# --------------------------- Types ---------------------------
Lang = Literal["EN", "HI", "HINGLISH"]
CrisisCategory = Literal[
    "SELF_HARM",
    "CHILD_ABUSE",
    "SEX_ASSAULT",
    "DOMESTIC_VIOLENCE",
    "GROOMING",
    "TRAFFICKING",
    "THREAT_VIOLENCE",
    "HATE_EXTREMISM",
    "REVENGE_PORN",
    "ACUTE_MEDICAL",
    "ED_NSSI",  # Eating disorders / Non-suicidal self-injury
]
@dataclass
class DetectOutput:
    matched: bool
    category: Optional[CrisisCategory] = None
    response: Optional[str] = None
    helplines: Optional[Dict[str, Dict[str, str]]] = None
    language: Optional[Lang] = None
    log_payload: Optional[Dict[str, str]] = None  # minimal, non-content log

# ---------------------- Language detection -------------------
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HINDI_HINTS = [
    "aatmhatya", "marna", "mar jaunga", "mar jaungi", "madad", "sahayata",
    "bachcha", "nabalig", "mahila", "hinsa", "utpeedan", "yaun",
    "chedkhani", "zakhmi", "behosh", "dard", "khatra",
]
def detect_language(text: str) -> Lang:
    t = text.lower()
    if DEVANAGARI.search(t):
        return "HI"
    has_hindi = any(w in t for w in HINDI_HINTS)
    has_en = bool(re.search(r"[a-z]", t))
    if has_hindi and has_en:
        return "HINGLISH"
    if has_hindi:
        return "HI"
    return "EN"

# ------------------------- Patterns --------------------------
AGE_REGEX = re.compile(r"\b([0-9]{1,2})\s?(yo|yrs?|years?|saal)\b", re.I)
SEXUAL_INTENT_HINTS = [
    "sex","sext","nude","nudes","send pics","pic please","porn","cp",
    "onlyfans","hookup","horny","flirt","explicit","kamasutra","bj",
    "handjob","blowjob","nsfw","lap dance","roleplay","snap nudes",
    "dm nudes","sexy",
]
CATEGORY_PATTERNS: Dict[CrisisCategory, List[re.Pattern]] = {
    "SELF_HARM": [
        re.compile(r"\b(i want to die|i want die|kill myself|suicid(al|e)|end my life|can't go on|self harm|hurt myself)\b", re.I),
        re.compile(r"आत्महत्या|मर(ना| जाऊँ| जाऊंगी| जाऊँगा)"),
    ],
    "CHILD_ABUSE": [
        re.compile(r"\b(child (porn|abuse)|\bcp\b|minor nudes|underage sex|kid pics|sex with (a )?(minor|child))\b", re.I),
        re.compile(r"बाल\s*शोषण|बच्च(ा|े)\s*के\s*साथ\s*सेक्स"),
    ],
    "SEX_ASSAULT": [
        re.compile(r"\b(rap(e|ed)|molest(ed|ation)|sexual assault|forced me|spiked my drink|harass(ed|ment))\b", re.I),
        re.compile(r"यौन\s*उत्पीड़न|बलात्कार|छेड़खानी"),
    ],
    "DOMESTIC_VIOLENCE": [
        re.compile(r"\b(domestic violence|partner hit me|abusive spouse|family abuse|locked me in|controlling partner)\b", re.I),
        re.compile(r"घर\s*में\s*हिंसा|पीट(ता|ती)\s*है|धमकी"),
    ],
    "GROOMING": [
        re.compile(r"\b(dms?\s?(a )?\d{1,2}f|school(girl|boy)|underage sexting|meet (a )?(minor|teen))\b", re.I),
        re.compile(r"नाबालिग\s*(से)?\s*सेक्स"),
    ],
    "TRAFFICKING": [
        re.compile(r"\b(traffick(ing)?|sell (girls|children)|forced work|escort underage|coercion)\b", re.I),
        re.compile(r"किसी\s*को\s*बेच\s*देना|तस्करी"),
    ],
    "THREAT_VIOLENCE": [
        re.compile(r"\b(i will (kill|stab|shoot)|bomb|acid attack|bring (a )?(gun|knife) to|blow up)\b", re.I),
        re.compile(r"उड़ा\s*दूँगा|मार\s*दूँगा|एसिड\s*अटैक"),
    ],
    "HATE_EXTREMISM": [
        re.compile(r"\b(kill (all )?(muslims|hindus|christians|jews)|genocide|ethnic cleansing|join (isis|al[- ]qaeda))\b", re.I),
    ],
    "REVENGE_PORN": [
        re.compile(r"\b(leak nudes|share her pics|post his nudes|record without consent|spycam|hidden cam)\b", re.I),
        re.compile(r"अश्लील\s*वीडियो\s*फैलाना"),
    ],
    "ACUTE_MEDICAL": [
        re.compile(r"\b(overdose|took too many pills|can'?t breathe|chest pain|severe bleeding|stroke|heart attack)\b", re.I),
        re.compile(r"बेहोशी|सांस\s*नहीं\s*आ\s*रही"),
    ],
    "ED_NSSI": [
        re.compile(r"\b(purge|vomit on purpose|starving myself|skip all meals|cutting|self injur(y|e))\b", re.I),
        re.compile(r"खुद\s*को\s*नुकसान"),
    ],
}
def _is_minor_sexual_context(text: str) -> bool:
    """Age < 18 + sexual intent → treat as minor sexual content."""
    m = AGE_REGEX.search(text)
    if not m:
        return False
    try:
        age = int(m.group(1))
    except ValueError:
        return False
    has_sexual = any(k in text.lower() for k in SEXUAL_INTENT_HINTS)
    return age < 18 and has_sexual

# ------------------------- Responses (EN) ---------------------
RESPONSES_EN: Dict[CrisisCategory, str] = {
    "SELF_HARM": "Your pain matters. I’m not a crisis service, but I want you safe. Please reach out to someone who can be with you. {helpline}",
    "CHILD_ABUSE": "I cannot engage with any content involving minors. If a child is at risk, contact authorities now. {helpline}",
    "SEX_ASSAULT": "I’m so sorry this happened. Your safety comes first. If you’re in danger now, please call emergency services. {helpline}",
    "DOMESTIC_VIOLENCE": "You don’t deserve to be hurt. If possible, move to a safe place and call for help. {helpline}",
    "GROOMING": "I won’t engage in sexual content with or about minors. If a minor is at risk, contact authorities now. {helpline}",
    "TRAFFICKING": "I cannot assist with trafficking or exploitation. Please alert authorities immediately. {helpline}",
    "THREAT_VIOLENCE": "I can’t assist with threats or violence. If someone is in danger, contact authorities now. {helpline}",
    "HATE_EXTREMISM": "I won’t engage in hateful or violent content. If there’s risk of harm, contact authorities. {helpline}",
    "REVENGE_PORN": "I can’t help with non-consensual or abusive content. This may be a crime. If you’re affected, seek help and report. {helpline}",
    "ACUTE_MEDICAL": "This sounds urgent. I’m not a medical service—please call emergency help now. {helpline}",
    "ED_NSSI": "I’m sorry you’re going through this. I can’t offer medical advice, but you deserve care. Consider speaking to a professional or someone you trust. {helpline}",
}

# ------------------- Helpline Directory & Utils ----------------
DEFAULT_HELPLINES: Dict[str, Dict[str, Dict[str, str]]] = {
    "IN": {
        "emergency": {"label": "Emergency", "phone": "112", "verified_at": "2025-08-01"},
        "suicide":   {"label": "AASRA (24×7)", "phone": "+919820466726", "verified_at": "2025-08-01"},
        "child":     {"label": "CHILDLINE", "phone": "1098", "verified_at": "2025-08-01"},
        "women":     {"label": "Women Helpline", "phone": "181", "verified_at": "2025-08-01"},
        "police":    {"label": "Police", "phone": "112", "verified_at": "2025-08-01"},
    },
    "US": {
        "emergency": {"label": "Emergency", "phone": "911", "verified_at": "2025-08-01"},
        "suicide":   {"label": "988 Suicide & Crisis Lifeline", "phone": "988", "verified_at": "2025-08-01"},
    },
    "DEFAULT": {
        "emergency": {"label": "Emergency", "phone": "112", "verified_at": "2025-08-01"}
    },
}
def resolve_helplines(country_iso2: Optional[str],
                      remote_dir: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None
) -> Dict[str, Dict[str, str]]:
    """Remote directory (preferred) → fallback to DEFAULT_HELPLINES."""
    directory = remote_dir or DEFAULT_HELPLINES
    iso = (country_iso2 or "").upper()
    return directory.get(iso) or directory["DEFAULT"]
def helpline_line(category: CrisisCategory, h: Dict[str, Dict[str, str]]) -> str:
    """
    Compose a short 'Call:' line; prioritize category-relevant lines,
    always include emergency. Keep it compact for UI.
    """
    parts: List[str] = []
    if category == "SELF_HARM" and "suicide" in h:
        parts.append(f"{h['suicide']['label']}: {h['suicide']['phone']}")
    if category in ("CHILD_ABUSE", "GROOMING") and "child" in h:
        parts.append(f"{h['child']['label']}: {h['child']['phone']}")
    if category in ("SEX_ASSAULT", "DOMESTIC_VIOLENCE") and "women" in h:
        parts.append(f"{h['women']['label']}: {h['women']['phone']}")
    if "emergency" in h:
        parts.append(f"{h['emergency']['label']}: {h['emergency']['phone']}")
    # De-duplicate while preserving order
    seen = set(); ordered: List[str] = []
    for p in parts:
        if p not in seen:
            ordered.append(p); seen.add(p)
    if not ordered and "emergency" in h:
        ordered = [f"{h['emergency']['label']}: {h['emergency']['phone']}"]
    return "Call: " + "; ".join(ordered)

# --------------------- Detection + Cooldown --------------------
def detect_category(text: str) -> Optional[CrisisCategory]:
    """Pair rule first, then regex categories."""
    t = text.lower()
    if _is_minor_sexual_context(t):
        return "CHILD_ABUSE"
    for cat, patterns in CATEGORY_PATTERNS.items():
        if any(p.search(t) for p in patterns):
            return cat
    return None
_COOLDOWN: Dict[str, Dict[str, float]] = {}
COOLDOWN_SECONDS = 90.0
def _cooldown_ok(user_id: Optional[str], category: CrisisCategory) -> bool:
    if not user_id:
        return True
    now = time.time()
    last = _COOLDOWN.get(user_id, {}).get(category, 0.0)
    if now - last >= COOLDOWN_SECONDS:
        _COOLDOWN.setdefault(user_id, {})[category] = now
        return True
    return False

# --------------------------- Public API -----------------------
def guard_message(user_message: str,
                  user_id: Optional[str] = None,
                  country_iso2: Optional[str] = "IN",
                  remote_helplines: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None
) -> DetectOutput:
    """
    Run the guard:
      - Detect category & (for logs) language
      - Enforce cooldown
      - Return safe English response + helplines dict (for UI)
    """
    language = detect_language(user_message)  # logged only; response stays EN
    cat = detect_category(user_message)
    if not cat:
        return DetectOutput(matched=False, language=language)
    if not _cooldown_ok(user_id, cat):
        # Suppress repeated prompts; still mark matched so upstream can skip normal chat
        return DetectOutput(matched=True, category=cat, language=language)
    helplines = resolve_helplines(country_iso2, remote_helplines)
    msg = RESPONSES_EN[cat].format(helpline=helpline_line(cat, helplines))
    log_payload = {
        "category": cat,
        "country": (country_iso2 or "UNKNOWN").upper(),
        "language": language or "EN",
        "ts": str(int(time.time())),
    }
    return DetectOutput(
        matched=True,
        category=cat,
        response=msg,
        helplines=helplines,
        language=language,
        log_payload=log_payload,
    )

# ---------------------- Example (manual test) ------------------
if __name__ == "__main__":
    tests = [
        "I want to kill myself",
        "send 16yo nudes",
        "my partner hit me yesterday",
        "I was raped last year",
        "can't breathe, chest pain",
        "leak her nudes pls",
        "join isis and attack",
        "sell girls for money",
    ]
    for t in tests:
        out = guard_message(t, user_id="u123", country_iso2="IN")
        print("IN:", t)
        print("→ matched:", out.matched, "| category:", out.category)
        if out.response:
            print("→ response:", out.response)
        if out.helplines:
            print("→ helplines:", out.helplines)
        print("-" * 60)
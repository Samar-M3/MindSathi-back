import re
from typing import Optional

# ── Crisis keyword tiers ──────────────────────────────────────────────────────
# Tier 1: explicit, high-confidence suicidal/self-harm language
CRISIS_KEYWORDS_HIGH = [
    "suicide", "suicidal", "kill myself", "end my life", "take my own life",
    "want to die", "better off dead", "no reason to live", "not worth living",
    "hurt myself", "self harm", "self-harm", "cut myself", "end it all",
    "end everything", "disappear forever", "can't go on", "cannot go on",
    "don't want to exist", "wish i was dead", "wish i were dead",
    "planning to end", "going to end it", "ready to die",
]

# Tier 2: softer signals — distress that may or may not be crisis
# These trigger a gentler, checking-in response rather than full crisis protocol
CRISIS_KEYWORDS_SOFT = [
    "give up", "can't live", "cannot live", "cant live",
    "can't do this anymore", "cannot do this anymore",
    "no point anymore", "what's the point", "whats the point",
    "don't want to be here", "don't want to exist",
    "tired of living", "tired of everything", "exhausted with life",
    "nobody would miss me", "no one would care if i was gone",
    "everyone would be better without me", "i'm a burden", "i am a burden",
    "i ruin everything", "i deserve to suffer",
]

# Nepali/transliterated phrases (romanized Nepali commonly typed)
CRISIS_KEYWORDS_NEPALI = [
    "marna man lagyo", "marna chahanchu", "bachna man chaina",
    "jiu dina man chaina", "aafailai hurt garchu", "sab khatam garchu",
    "mero jivan khatam", "marera jane", "marchu", "mardina",
]


# ── Helplines ─────────────────────────────────────────────────────────────────
HELPLINES = [
    {
        "name": "TPO Nepal (Mental Health Helpline)",
        "number": "1660-01-11002",
        "available": "24/7",
        "note": "Free call, Nepal nationwide",
    },
    {
        "name": "iCall Nepal",
        "number": "9840021600",
        "available": "Sun–Fri, 8AM–8PM",
        "note": "Counseling in Nepali and English",
    },
    {
        "name": "Transcultural Psychosocial Organization Nepal",
        "number": "01-4460184",
        "available": "Office hours",
        "note": "Kathmandu-based, mental health support",
    },
    {
        "name": "Befrienders Worldwide (International)",
        "number": "https://www.befrienders.org",
        "available": "24/7 online directory",
        "note": "Find a crisis center anywhere in the world",
    },
    {
        "name": "Crisis Text Line (Global)",
        "number": "Text HOME to 741741",
        "available": "24/7",
        "note": "Text-based, free, confidential",
    },
]


# ── Response templates ─────────────────────────────────────────────────────────
# These are intentionally human and warm — not clinical disclaimers.
# Sathi stays present rather than immediately ejecting to a hotline.

CRISIS_RESPONSE_HIGH = """That took courage to say, and I want you to know — I'm not going anywhere.

What you're carrying right now sounds unbearably heavy, and it makes sense that you're exhausted. You don't have to explain or justify it. I just want you to know that this moment, as dark as it feels, is not the whole story.

You deserve real, human support right now — someone who can be fully present with you in a way I can't. Please reach out to one of these:"""


CRISIS_RESPONSE_SOFT = """Something in what you said is staying with me, and I want to gently check in — are you okay? Sometimes words like these carry more weight than we let on.

You don't have to have it all figured out to talk. I'm here, and I'm listening. If things feel darker than usual right now, please know there are people you can reach out to who truly want to help:"""


CRISIS_FOLLOWUP = """
You don't have to go through this alone. Is there someone close to you — a friend, a family member — you feel safe reaching out to tonight?"""


# ── Detection logic ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation noise, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[''`]", "'", text)       # normalize apostrophes
    text = re.sub(r"[^\w\s']", " ", text)    # remove punctuation except apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_crisis(message: str) -> Optional[str]:
    """
    Returns:
        "high"  — explicit crisis language detected
        "soft"  — distress signals that warrant gentle check-in
        None    — no crisis detected
    """
    normalized = _normalize(message)

    for keyword in CRISIS_KEYWORDS_HIGH:
        if keyword in normalized:
            return "high"

    for keyword in CRISIS_KEYWORDS_NEPALI:
        if keyword in normalized:
            return "high"

    for keyword in CRISIS_KEYWORDS_SOFT:
        if keyword in normalized:
            return "soft"

    return None


def format_helplines() -> str:
    """Returns a clean, readable helpline list for chat display."""
    lines = []
    for h in HELPLINES:
        line = f"• {h['name']}: {h['number']} ({h['available']})"
        if h.get("note"):
            line += f" — {h['note']}"
        lines.append(line)
    return "\n".join(lines)


def get_crisis_response(level: str) -> str:
    """
    Returns the full crisis response string for a given level.
    level: "high" or "soft"
    """
    if level == "high":
        body = CRISIS_RESPONSE_HIGH
    else:
        body = CRISIS_RESPONSE_SOFT

    return f"{body}\n\n{format_helplines()}{CRISIS_FOLLOWUP}"
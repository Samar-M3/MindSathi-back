import re
from typing import Optional

import google.generativeai as genai
import os

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
    "baacha rahanu pardaina", "jiunu man chaina", "sab chhadna man lagyo",
]

# Nepali soft signals
CRISIS_KEYWORDS_NEPALI_SOFT = [
    "dherai thakeko", "runa man lagyo", "eklo feel huncha",
    "koi chhaina", "khasai farak pardaina", "hराउन man lagyo",
    "sab thakayo", "aba sakdina", "hope chaina",
]


# ── Semantic crisis detection prompt ─────────────────────────────────────────
# Used when keyword match is ambiguous — asks Gemini to assess severity
SEMANTIC_CRISIS_PROMPT = """You are a crisis detection system for a mental health app.

Assess the following message and return ONLY one of these three words — nothing else:
- HIGH   (explicit suicidal ideation, self-harm intent, or imminent danger)
- SOFT   (emotional distress, hopelessness, or passive thoughts about not wanting to exist — but no explicit intent)  
- NONE   (general emotional difficulty, venting, or no crisis signal)

Be conservative: if unsure between SOFT and NONE, return SOFT.
If unsure between HIGH and SOFT, return HIGH.

Message: {message}"""


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
CRISIS_RESPONSE_HIGH = """that took real courage to say, and I want you to know — I'm not going anywhere.

what you're carrying right now sounds unbearably heavy. it makes sense you're exhausted. you don't have to explain or justify it.

but I want to be honest with you — right now, you deserve more than I can give. you deserve a real human voice, someone who can truly be there with you in this moment.

please reach out to one of these:"""


# Warmer, more personal soft response — feels less like a protocol, more like a friend noticing
CRISIS_RESPONSE_SOFT = """hey — something in what you just said is sitting with me, and I want to check in properly.

are you doing okay? not "fine" okay — actually okay.

sometimes those words carry more weight than we let on, and I don't want to just gloss over it. if things are feeling darker than usual lately, you don't have to carry that alone.

there are people you can talk to who genuinely want to help:"""


CRISIS_FOLLOWUP = """
you don't have to go through this alone. is there someone close to you — a friend, a family member — you feel safe reaching out to tonight?"""


# ── Detection logic ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation noise, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_detect(message: str) -> Optional[str]:
    """Fast keyword-based detection. Returns 'high', 'soft', or None."""
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

    for keyword in CRISIS_KEYWORDS_NEPALI_SOFT:
        if keyword in normalized:
            return "soft"

    return None


# Phrases that sound distressing but are almost always casual/metaphorical
_FALSE_POSITIVE_GUARDS = [
    "i want to kill this bug",
    "kill this project",
    "kill it at",
    "killing it",
    "dying of laughter",
    "dying to see",
    "dead tired",
    "die laughing",
    "i could die",         # e.g. "i could die of embarrassment"
    "i'm dead",            # slang for finding something funny
    "literally dying",     # slang
]

def _is_likely_false_positive(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in _FALSE_POSITIVE_GUARDS)


def _semantic_detect(message: str) -> Optional[str]:
    """
    Uses Gemini to semantically assess crisis level for ambiguous messages.
    Falls back to None if API call fails — fail open (don't block safe messages).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.0,      # deterministic for safety checks
                max_output_tokens=10,
            ),
        )
        prompt = SEMANTIC_CRISIS_PROMPT.format(message=message)
        response = model.generate_content(prompt)
        result = response.text.strip().upper()

        if result == "HIGH":
            return "high"
        elif result == "SOFT":
            return "soft"
        else:
            return None
    except Exception as e:
        print(f"Semantic crisis detection failed: {e}")
        return None  # fail safe — don't block the message


def detect_crisis(message: str) -> Optional[str]:
    """
    Multi-layer crisis detection:
    1. False positive guard (skip if clearly metaphorical)
    2. Fast keyword matching
    3. Semantic AI detection for ambiguous messages

    Returns:
        "high"  — explicit crisis language detected
        "soft"  — distress signals that warrant gentle check-in
        None    — no crisis detected
    """
    # Layer 0 — false positive guard
    if _is_likely_false_positive(message):
        return None

    # Layer 1 — fast keyword check
    keyword_result = _keyword_detect(message)
    if keyword_result == "high":
        return "high"

    # Layer 2 — semantic check for messages that keywords missed OR flagged as soft
    # This catches phrases like "I just feel like there's no point anymore" or
    # "I don't think anyone would notice if I wasn't here"
    # Also upgrades soft keyword hits if semantic check returns high
    semantic_result = _semantic_detect(message)

    if semantic_result == "high":
        return "high"
    if semantic_result == "soft" or keyword_result == "soft":
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


def get_crisis_response(level: str, doctors: list[dict] | None = None) -> str:
    """
    Returns the full crisis response string for a given level.
    level: "high" or "soft"
    doctors: optional list of nearby clinic dicts from places.py
    """
    if level == "high":
        body = CRISIS_RESPONSE_HIGH
    else:
        body = CRISIS_RESPONSE_SOFT

    response = f"{body}\n\n{format_helplines()}"

    # Append nearby doctors section if available
    if doctors:
        from app.services.places import format_doctors_for_chat
        doctors_text = format_doctors_for_chat(doctors)
        if doctors_text:
            response += f"\n\n{doctors_text}"

    response += CRISIS_FOLLOWUP
    return response
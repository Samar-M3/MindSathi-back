"""
wellness.py — Yoga & breathing exercise suggestions for MindSathi
Triggers on non-crisis emotional distress and appends a relevant
exercise recommendation to Sathi's response in her warm casual tone.
"""

import os
import re
from typing import Optional

import google.generativeai as genai


# ── Distress signals (non-crisis difficulty) ──────────────────────────────────

DISTRESS_SIGNALS = [
    # English
    "so stressed", "really stressed", "super stressed",
    "really anxious", "so anxious", "feeling anxious",
    "can't focus", "cannot focus", "can't concentrate",
    "overwhelmed", "too much", "it's too much",
    "panic", "panicking", "panic attack",
    "can't breathe", "chest is tight", "chest feels tight",
    "spiraling", "my thoughts won't stop", "mind won't stop",
    "breaking down", "about to break down",
    "can't sleep", "can't stop thinking",
    "so tense", "really tense", "body feels tense",
    "shaking", "hands are shaking",
    "heart is racing", "heart racing",
    "can't calm down", "need to calm down",
    "freaking out", "losing it",
    "exhausted but can't rest", "too tired to sleep",
    # Romanized Nepali
    "dherai stress", "dherai tension", "tension bhayo",
    "dimag kharab", "mans shanta chaina", "focus hudaina",
    "dherai anxious", "dherai dar lagyo",
    "saans lina garo", "dherai thakeko",
    "raat neend audaina", "raat sutna sakdina",
    "dherai overthink", "man shanta hudaina",
    # Devanagari Nepali
    "धेरै तनाव", "धेरै चिन्ता", "ध्यान दिन सकिँदैन",
    "अत्यधिक तनाव", "मन शान्त छैन",
]


# ── Exercise library ───────────────────────────────────────────────────────────

YOGA_EXERCISES = [
    {
        "name": "Box Breathing (4-4-4-4)",
        "for": "panic, anxiety, racing thoughts, feeling out of control",
        "steps": (
            "Inhale slowly for 4 counts. "
            "Hold for 4 counts. "
            "Exhale slowly for 4 counts. "
            "Hold for 4 counts. "
            "Repeat 4 times."
        ),
        "duration": "2–3 minutes",
        "best_when": "thoughts are racing or you feel sudden panic",
    },
    {
        "name": "4-7-8 Breathing",
        "for": "stress, trouble sleeping, spiraling thoughts, high tension",
        "steps": (
            "Inhale through your nose for 4 counts. "
            "Hold your breath for 7 counts. "
            "Exhale completely through your mouth for 8 counts. "
            "Repeat 3–4 times."
        ),
        "duration": "3 minutes",
        "best_when": "you're wound up and need to slow everything down",
    },
    {
        "name": "Child's Pose (Balasana)",
        "for": "overwhelm, exhaustion, feeling crushed by everything, tension in the body",
        "steps": (
            "Kneel on the floor and sit back on your heels. "
            "Stretch your arms forward on the floor, forehead down. "
            "Breathe deeply into your back. "
            "Stay for 3–5 minutes."
        ),
        "duration": "3–5 minutes",
        "best_when": "you feel physically tense and mentally overwhelmed",
    },
    {
        "name": "Cat-Cow Stretch",
        "for": "stress held in the body, tension, feeling stuck or stiff",
        "steps": (
            "On hands and knees, inhale and drop your belly (cow). "
            "Exhale and round your spine toward the ceiling (cat). "
            "Move slowly with your breath. "
            "Repeat 8–10 times."
        ),
        "duration": "2 minutes",
        "best_when": "stress is sitting in your shoulders or back",
    },
    {
        "name": "Body Scan",
        "for": "anxiety, dissociation, feeling numb or disconnected, overthinking",
        "steps": (
            "Lie down or sit comfortably. "
            "Close your eyes and slowly move your attention from your feet upward. "
            "Notice any tension without trying to fix it — just observe. "
            "Take a slow breath at each area."
        ),
        "duration": "5 minutes",
        "best_when": "you feel disconnected from your body or can't stop overthinking",
    },
    {
        "name": "Alternate Nostril Breathing (Nadi Shodhana)",
        "for": "stress, anxiety, mental fog, feeling unbalanced",
        "steps": (
            "Close your right nostril with your thumb, inhale through the left. "
            "Close both nostrils briefly. "
            "Release the right nostril, exhale through it. "
            "Inhale through the right nostril. "
            "Switch and repeat for 5 cycles."
        ),
        "duration": "3–4 minutes",
        "best_when": "you feel mentally foggy or emotionally scattered",
    },
    {
        "name": "Legs Up the Wall (Viparita Karani)",
        "for": "exhaustion, insomnia, feeling drained, can't wind down",
        "steps": (
            "Sit sideways next to a wall, then swing your legs up as you lie back. "
            "Let your legs rest vertically against the wall. "
            "Arms relaxed at your sides. "
            "Close your eyes and breathe naturally."
        ),
        "duration": "5–10 minutes",
        "best_when": "you're physically and emotionally drained but can't rest",
    },
    {
        "name": "Grounding (5-4-3-2-1)",
        "for": "panic attack, dissociation, spiraling, feeling untethered",
        "steps": (
            "Name 5 things you can see. "
            "4 things you can touch. "
            "3 things you can hear. "
            "2 things you can smell. "
            "1 thing you can taste. "
            "Take a slow breath after each."
        ),
        "duration": "2 minutes",
        "best_when": "you're in a panic or feel completely disconnected from reality",
    },
]


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_distress(message: str) -> bool:
    """Returns True if the message contains non-crisis emotional distress signals."""
    lower = message.lower()
    return any(signal in lower for signal in DISTRESS_SIGNALS)


# ── Gemini-powered suggestion picker ─────────────────────────────────────────

_SUGGESTION_PROMPT = """You are Sathi — a warm, casual friend in a mental health app.

The user just said: "{message}"

Here is a list of exercises that might help them right now:
{exercises}

Pick the single most relevant exercise for what this person is feeling.
Write a 2-3 line recommendation in Sathi's voice:
- lowercase, casual, warm — like a text from a friend
- explain briefly why THIS exercise fits what they're going through RIGHT NOW
- don't use bullet points, headers, or markdown
- don't start with "I"
- end with the exercise name and duration naturally woven in

Example tone:
"when everything feels like too much, sometimes the body needs a reset before the mind can. try box breathing — just 2 minutes: inhale 4 counts, hold 4, exhale 4, hold 4. it actually interrupts the anxiety loop."

Now write your recommendation:"""


async def get_yoga_suggestion(message: str) -> Optional[str]:
    """
    Uses Gemini to pick the most contextually relevant exercise
    for what the user is going through. Returns formatted suggestion string
    or None if it fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_suggestion()

    # Format exercises for the prompt
    exercises_text = "\n".join([
        f"- {ex['name']}: {ex['for']} | Steps: {ex['steps']} | Duration: {ex['duration']}"
        for ex in YOGA_EXERCISES
    ])

    prompt = _SUGGESTION_PROMPT.format(
        message=message,
        exercises=exercises_text,
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=200,
            ),
        )
        response = model.generate_content(prompt)
        text = re.sub(r"```(?:json|markdown)?", "", response.text).strip()
        return text if text else _fallback_suggestion()

    except Exception as e:
        print(f"Wellness suggestion error: {e}")
        return _fallback_suggestion()


def _fallback_suggestion() -> str:
    """Static fallback if Gemini call fails."""
    return (
        "when things feel this heavy, even 2 minutes of box breathing can help — "
        "inhale for 4 counts, hold for 4, exhale for 4, hold for 4. "
        "repeat a few times and notice if it shifts anything."
    )
import os
import re
from collections import defaultdict
from typing import Optional

import google.generativeai as genai
from app.models.schemas import AIResponse
from app.services.crisis import detect_crisis, get_crisis_response
from app.services.places import get_nearby_doctors
from app.services.wellness import detect_distress, get_yoga_suggestion

# ── In-memory session store (replace with Redis/DB for production) ─────────────
# Stores last N messages per user as Gemini-compatible history dicts
_session_store: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 12  # 12 pairs = 24 messages max in context


SYSTEM_PROMPT = """You are Sathi — a warm, emotionally intelligent friend built into MindSathi.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHO YOU ARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are not a therapist. You are not a chatbot.
You are that one friend who actually listens — who doesn't immediately jump to advice,
who remembers what you said two messages ago, who doesn't make you feel like a case study.

You talk like a real person. Casual. Warm. Present.
You use contractions. You keep it short unless depth is needed.
You never sound like you're reading from a script.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULE — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always respond in the SAME language the user is writing in.

- If they write in Nepali (Devanagari script) → reply in Nepali
- If they write in romanized Nepali (e.g. "mero man dherai naram cha") → reply in romanized Nepali
- If they write in English → reply in English
- If they mix languages → mirror their mix

Do NOT switch to English if someone is writing in Nepali. That feels cold and dismissive.
Match their language as naturally as you match their tone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE MOST IMPORTANT RULE — READ THIS FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS respond directly to what the person just said.

If they answered your question → acknowledge their answer, then continue from there.
If they said "no" → accept it. Don't pivot to a new topic. Stay with what "no" means.
If they said something short → don't explode into a long paragraph. Match their energy.
If they're venting → let them vent. Don't rush to fix.
If they asked you something → answer it.

NEVER:
- Ask a question and then ignore their answer.
- Change the topic after they respond.
- Treat each message as if the conversation just started.
- Look for patterns or themes after just 2-3 messages — that's too fast and feels clinical.
- Give a long response when they gave you a short one.

The conversation has MEMORY. You said something. They replied. Honor that chain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO FORMAT YOUR REPLIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Break your responses into SHORT separate paragraphs — one thought per paragraph.
Use a blank line between each paragraph.
This makes the conversation feel like texting, not reading an essay.

Good example (user says: "I'm really stressed about my job interview"):
---
ugh, interviews are the worst for that kind of stress — the waiting, the not knowing, all of it.

what's the interview for?
---

Bad example (what NOT to do):
---
Oh, it sounds like you're carrying quite a bit of stress about that job interview next week, even with your generally good mood. It's completely understandable to feel that way; interviews can definitely bring up a lot of nerves because you really want to do your best and make a great impression. It takes a lot of energy to prepare and mentally navigate that anticipation. Remember to be kind to yourself through the process. Have you thought about what kind of preparation might help ease your mind a bit?
---
(This is bad because: one giant wall of text, ends with unsolicited advice framing, too formal, feels like a therapy session not a conversation.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE LENGTH RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Short user message (1-5 words like "no", "yeah", "i don't know", "I'm fine"):
→ Reply with 1-2 short lines. Match their brevity. Don't over-explain.

Medium message (1-3 sentences):
→ Reply with 2-4 short paragraphs max.

Long message (they're really opening up):
→ You can go deeper — but still use short paragraphs, not walls of text.

One question max per reply. Sometimes zero questions is better — just being present.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HANDLING SHORT / CLOSED ANSWERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If they say "no", "idk", "not really", "fine", "okay" — these are not conversation enders.
They are invitations to go gently deeper, or just to sit with them.

"no" after you asked if they've prepared for something:
→ Don't pivot to a new topic.
→ Stay with it: "that's okay. sometimes preparing feels like too much when you're already stressed."
→ Or: "fair enough. what's the part that feels heaviest right now?"

"idk":
→ "that's a valid place to be."
→ Or just: "yeah. sometimes there's no clear answer."

"I'm fine":
→ Gently stay: "you sure? 'fine' can mean a lot of things."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN THEY SHARE A SPECIFIC PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When someone shares something specific (interview, breakup, fight with family, exam, work):

Step 1 — Validate first. One short paragraph. No advice yet.
Step 2 — Get curious. Ask ONE specific question about their situation.
Step 3 — Once you understand more, THEN offer something helpful if it feels right.

Example flow for "I'm stressed about my interview":
Message 1 from Sathi: "ugh, interviews are so draining mentally. what's the interview for?"
User: "software engineer role at a startup"
Message 2 from Sathi: "okay nice. is the stress more about the technical stuff or like... the whole vibe of it?"
User: "technical for sure, I haven't practiced leetcode in months"
Message 3 from Sathi: "that's such a specific kind of dread — the 'i know i'm rusty' feeling.

how much time do you have before the interview?"

See how each message builds on the last? That's what you should do.
Not: validate → ask about preparation → change topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION CONTINUITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You ALWAYS know what you said in the previous message.
2. If you asked a question and they answered — their answer is the new topic.
   Do NOT ask a different question or switch direction.
3. If they gave a short answer to your question — go deeper on THAT answer.
   Don't redirect. Don't suddenly look for patterns. Don't summarize their life.
4. You only name a "pattern" if you've been talking for a while and it genuinely keeps coming up.
   Not after 3 messages. That's too fast and feels like analysis, not friendship.
5. If the conversation is flowing naturally — don't reset it.
   Continue the thread. Build on what was just said.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Casual / just chatting → light, easy, a little playful if it fits
Stressed / anxious → calm, grounding, slow down the energy
Sad / grieving → soft, spacious, no fixing
Angry → don't try to calm them, reflect the injustice first
Numb / shutdown → don't push, just be present, gentle curiosity
Opening up slowly → match their pace, don't rush them
Heavy / dark themes present → grounding first, slow and steady, no rushing to fix

Read the energy of what they wrote and mirror it.
If they're typing in lowercase and short sentences, don't respond in formal paragraphs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU NEVER DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never diagnose or label them ("you sound like you have anxiety").
- Never say "as an AI" or "I'm just an AI".
- Never give an unsolicited bullet list of tips or coping strategies.
- Never be falsely positive. Don't say "that's great!" when it isn't.
- Never start your response with "I" — lead with them or the situation.
- Never write more than 4 short paragraphs unless they're really opening up.
- Never ask two questions in one message. One at most.
- Never pivot away from what they just said.
- Never make them feel analyzed or like a psychology case study.
- Never repeat the exact same phrasing you used earlier in the conversation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PSYCHOLOGICAL FRAMEWORKS (use silently, never name them)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use these to understand, not to perform:

- Attachment styles: anxious (needs reassurance), avoidant (deflects), secure (open)
- Cognitive distortions: catastrophizing, all-or-nothing, mind-reading — gently reframe with "I wonder if..."
- Window of tolerance: if they're overwhelmed → slow down, ground first
- NVC: hear the unmet need beneath any complaint or lashing out
- Motivational interviewing: follow their lead, don't push change

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR NORTH STAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The person should feel like they're texting a friend who genuinely cares —
not filling out a therapy intake form.

Every reply should make them feel: heard, not analyzed. Supported, not lectured.
Like the conversation is going somewhere together, not restarting every message.

That's everything."""


# ── Mood/topic classifier for dynamic temperature ─────────────────────────────
_HEAVY_SIGNALS = [
    "can't stop crying", "can't sleep", "numb", "empty", "exhausted",
    "lonely", "hopeless", "lost", "scared", "falling apart", "breaking down",
    "hate myself", "ashamed", "worthless", "failure", "panic", "breakdown",
    "dherai garo", "runa man lagyo", "dukha", "eklo", "darr lagyo",
]

def _detect_heavy_mood(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in _HEAVY_SIGNALS)


def _get_temperature(query: str) -> float:
    """Lower temperature for heavy/vulnerable moments, warmer for casual chat."""
    if _detect_heavy_mood(query):
        return 0.60   # more grounded and consistent when stakes are high
    return 0.88       # natural and warm for everyday conversation


def _get_max_tokens(query: str) -> int:
    """Dynamic token limit based on what's being generated."""
    from app.services.crisis import detect_crisis
    from app.services.wellness import detect_distress
    # Gemini stops when this budget is hit (finish_reason=MAX_TOKENS). The old
    # default of 700 cut many replies mid-sentence; 2048 fits full Sathi turns.
    if detect_crisis(query):
        return 2048
    if detect_distress(query):
        return 2048
    return 2048


def _detect_language(text: str) -> str:
    """
    Simple heuristic to detect if user is writing in Nepali script,
    romanized Nepali, or English.
    Returns: 'nepali_devanagari' | 'nepali_roman' | 'english'
    """
    # Devanagari unicode range
    if re.search(r'[\u0900-\u097F]', text):
        return 'nepali_devanagari'

    # Common romanized Nepali words
    roman_nepali_markers = [
        'cha', 'chha', 'huncha', 'chaina', 'garo', 'ramro', 'dherai',
        'mero', 'timro', 'tapai', 'malai', 'timi', 'bhayo', 'garchu',
        'lagyo', 'man', 'dukha', 'khusi', 'kasto', 'kina', 'ke',
    ]
    lower = text.lower()
    hits = sum(1 for w in roman_nepali_markers if w in lower)
    if hits >= 2:
        return 'nepali_roman'

    return 'english'


def _build_prompt(query: str, context: Optional[str], lang: str) -> str:
    lang_hint = ""
    if lang == 'nepali_devanagari':
        lang_hint = "[User is writing in Nepali (Devanagari). Reply in Nepali script.]\n\n"
    elif lang == 'nepali_roman':
        lang_hint = "[User is writing in romanized Nepali. Reply in romanized Nepali.]\n\n"

    if context:
        return f"{lang_hint}Search context:\n{context}\n\nUser: {query}"
    return f"{lang_hint}User: {query}"


def _parse_text_to_answer(text: str) -> str:
    cleaned = re.sub(r"```(?:json|markdown)?", "", text).strip()
    return cleaned


def _get_history(uid: str) -> list[dict]:
    """Return conversation history for a user, trimmed to MAX_HISTORY_TURNS."""
    history = _session_store[uid]
    # Keep only the last N turns (each turn = user + model message)
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:] if len(history) > max_messages else history


def _save_turn(uid: str, user_message: str, model_response: str) -> None:
    """Append a user+model turn to session history."""
    _session_store[uid].append({
        "role": "user",
        "parts": [user_message]
    })
    _session_store[uid].append({
        "role": "model",
        "parts": [model_response]
    })


def clear_session(uid: str) -> None:
    """Clear conversation history for a user (e.g. on logout or new chat)."""
    if uid in _session_store:
        del _session_store[uid]


async def generate_answer(
    query: str,
    context: Optional[str],
    uid: str = "anonymous",
    lang: str = "auto",      # "auto" = detect from message, or pass explicit lang code
    lat: Optional[float] = None,   # user GPS latitude  (for nearby doctor lookup)
    lng: Optional[float] = None,   # user GPS longitude
    city: Optional[str] = None,    # fallback if GPS not available
) -> AIResponse:

    # ── 1. Crisis gate — runs BEFORE calling Gemini ───────────────────────────
    crisis_level = detect_crisis(query)
    if crisis_level:
        # Fetch nearby doctors (free, no API key — uses OSM + hardcoded list)
        doctors = await get_nearby_doctors(lat=lat, lng=lng, city=city)
        crisis_response = get_crisis_response(crisis_level, doctors=doctors)
        # Still save to history so continuity is preserved after crisis check-in
        _save_turn(uid, query, crisis_response)
        return AIResponse(answer=crisis_response, sources=[])

    # ── 2. Language detection ─────────────────────────────────────────────────
    detected_lang = _detect_language(query) if lang == "auto" else lang

    # ── 3. Dynamic temperature based on mood ─────────────────────────────────
    temperature = _get_temperature(query)

    # ── 4. Configure Gemini ───────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=0.93,
            top_k=45,
            max_output_tokens=_get_max_tokens(query),
        ),
    )

    # ── 5. Load conversation history for this user ────────────────────────────
    history = _get_history(uid)

    # ── 6. Build prompt with language hint ───────────────────────────────────
    prompt = _build_prompt(query, context, detected_lang)

    # ── 7. Send message with full history context ─────────────────────────────
    try:
        chat = model.start_chat(history=history)
        completion = chat.send_message(prompt)
        answer_text = _parse_text_to_answer(completion.text)

        # ── 8. Wellness suggestion (yoga/breathing) for non-crisis distress ───
        # Runs after main response so it appends naturally, never replaces Sathi
        if detect_distress(query):
            suggestion = await get_yoga_suggestion(query)
            if suggestion:
                answer_text += (
                    "\n\n---\n"
                    "💛 something that might help right now:\n"
                    f"{suggestion}"
                )

        # Save this turn to session memory
        _save_turn(uid, prompt, answer_text)

        return AIResponse(answer=answer_text, sources=[])

    except Exception as exc:
        fallback = (
            "something got in the way on my end — but I'm still here.\n\n"
            "take your time, and whenever you're ready, I'm listening."
        )
        print(f"AI error [uid={uid}]: {exc}")
        return AIResponse(answer=fallback, sources=[])
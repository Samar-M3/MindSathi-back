import asyncio
import json
import os
import re
from datetime import datetime
from typing import Optional

import google.generativeai as genai
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.firebase_service import init_firebase

router = APIRouter()


class SessionSummarizeRequest(BaseModel):
    uid: str
    messages: list
    sessionId: str


SUMMARY_PROMPT = """
You are summarizing a private mental wellness conversation.
Be warm, human, and gentle - not clinical.

Analyze this conversation and return ONLY valid JSON with no other text:
{
  "title": "2-4 word theme of this session (e.g. 'Work stress and boundaries')",
  "summary": "2-3 warm sentences describing what was explored and how the person seemed. Write in second person: 'You talked about...'",
  "themes": ["theme1", "theme2", "theme3"],
  "copingStrategies": ["any strategies mentioned or that Sathi suggested - empty array if none"],
  "breakthrough": "one genuine positive insight or moment if there was one, otherwise null",
  "dominantMood": "single lowercase word describing the overall emotional tone",
  "gentleTakeaway": "one encouraging sentence for the user to carry forward after this session"
}

The conversation:
{transcript}
"""

FALLBACK_SUMMARY = {
    "title": "Your session",
    "summary": "You had a meaningful conversation with Sathi today.",
    "themes": [],
    "copingStrategies": [],
    "breakthrough": None,
    "dominantMood": "neutral",
    "gentleTakeaway": "Every conversation is a step forward.",
}


def _clean_messages(messages: list) -> list:
    """Keep only user/assistant messages and trim to last ~30 pairs."""
    clean = [m for m in messages or [] if (m or {}).get("role") in ("user", "assistant")]
    if len(clean) > 60:
        clean = clean[-60:]
    return clean


def _build_transcript(messages: list) -> str:
    parts = []
    for msg in messages:
        role = "You" if msg.get("role") == "user" else "Sathi"
        content = str(msg.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        return json.loads(match.group()) if match else {}
    except Exception:
        return {}


def _get_model() -> Optional[genai.GenerativeModel]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-2.5-flash")


async def _generate_summary(transcript: str) -> dict:
    model = _get_model()
    if not model or not transcript:
        return FALLBACK_SUMMARY

    prompt = SUMMARY_PROMPT.replace("{transcript}", transcript)
    try:
        resp = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 500},
        )
        data = _parse_json(resp.text or "")
        return {**FALLBACK_SUMMARY, **data}
    except Exception as exc:
        print(f"Session summary error: {exc}")
        return FALLBACK_SUMMARY


@router.post("/api/session/summarize")
async def summarize_session(request: SessionSummarizeRequest):
    """
    Generate a summary of the session using Gemini and persist to Firestore.
    """
    uid = request.uid or "anonymous"
    messages = request.messages or []
    session_id = request.sessionId or "default"

    if len(messages) < 4 or not uid:
        return FALLBACK_SUMMARY

    clean_messages = _clean_messages(messages)
    transcript = _build_transcript(clean_messages)

    summary = await _generate_summary(transcript)

    # Save to Firestore when available and not anonymous
    try:
        db = init_firebase()
        if db and uid != "anonymous":
            doc_ref = db.collection("users").document(uid).collection("sessions").document(session_id)
            doc_ref.set(
                {
                    **summary,
                    "createdAt": datetime.utcnow(),
                    "messageCount": len(messages),
                },
                merge=True,
            )
    except Exception as exc:
        print(f"Session Firestore save error: {exc}")

    return summary

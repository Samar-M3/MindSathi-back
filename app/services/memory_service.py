import os
import json
import asyncio
import re
from datetime import datetime
from typing import Optional

import google.generativeai as genai
from google.cloud.firestore_v1 import AsyncClient


def _get_firestore_client() -> Optional[AsyncClient]:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not cred_path or not os.path.exists(cred_path):
        return None
    try:
        return AsyncClient.from_service_account_json(cred_path)
    except Exception as e:
        print(f"Firestore init failed: {e}")
        return None


async def get_user_memory(uid: str) -> str:
    """
    Fetch users/{uid}/memories/main from Firestore.
    Return formatted warm context string or "" if none.
    """
    client = _get_firestore_client()
    if not client:
        return ""

    doc_ref = client.collection("users").document(uid).collection("memories").document("main")
    try:
        snap = await doc_ref.get()
    except Exception as e:
        print(f"Memory fetch error: {e}")
        return ""

    if not snap.exists:
        return ""

    data = snap.to_dict() or {}
    summary = data.get("summary", "")
    themes = ", ".join(data.get("recurringThemes", []) or [])
    breakthroughs = ", ".join(data.get("breakthroughs", []) or [])

    return (
        "This person has spoken with Sathi before. Here is what you know:\n\n"
        f"{summary}\n\n"
        f"Things that keep coming up: {themes or '—'}\n"
        f"Positive moments: {breakthroughs or '—'}\n\n"
        "Use this naturally. Only reference when relevant. Never list robotically."
    )


async def update_user_memory(uid: str, session_messages: list):
    """
    Merge existing memory with a new session transcript using Gemini,
    then write back to Firestore.
    """
    client = _get_firestore_client()
    if not client:
        return

    doc_ref = client.collection("users").document(uid).collection("memories").document("main")

    # 1. Fetch existing
    try:
        snap = await doc_ref.get()
        existing = snap.to_dict() if snap.exists else {}
    except Exception as e:
        print(f"Memory read error: {e}")
        return

    existing_summary = existing.get("summary", "")
    existing_themes = existing.get("recurringThemes", [])
    existing_breakthroughs = existing.get("breakthroughs", [])
    session_count = existing.get("sessionCount", 0)

    # 2. Build transcript
    clean_messages = [m for m in session_messages if m.get("role") in ("user", "assistant")]
    transcript_parts = []
    # keep last 40 turns (20 pairs)
    for m in clean_messages[-40:]:
        role = "User" if m.get("role") == "user" else "Sathi"
        transcript_parts.append(f"{role}: {m.get('content', '').strip()}")
    transcript = "\n".join(transcript_parts)

    # 3. Gemini call
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    prompt = (
        "You are maintaining a private memory file for a mental wellness AI.\n"
        f"EXISTING SUMMARY: {existing_summary}\n"
        f"EXISTING THEMES: {existing_themes}\n"
        f"EXISTING BREAKTHROUGHS: {existing_breakthroughs}\n"
        f"NEW SESSION:\n{transcript}\n\n"
        "Update the memory. Keep summary under 300 words, warm and human.\n"
        "Write as describing a person to a caring friend, not a medical file.\n"
        "Max 8 themes, max 8 breakthroughs.\n\n"
        'Respond ONLY with valid JSON:\n{"summary": "...", "recurringThemes": [...], "breakthroughs": [...]}'
    )

    try:
        resp = await asyncio.to_thread(model.generate_content, prompt, generation_config={"max_output_tokens": 512})
        text = re.sub(r"```(?:json)?", "", (resp.text or "")).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        summary = data.get("summary", existing_summary)
        themes = data.get("recurringThemes", existing_themes)
        breakthroughs = data.get("breakthroughs", existing_breakthroughs)
    except Exception as e:
        print(f"Memory merge error: {e}")
        return

    # 4. Write back
    try:
        await doc_ref.set({
            "summary": summary,
            "recurringThemes": themes,
            "breakthroughs": breakthroughs,
            "lastUpdated": datetime.utcnow().isoformat(),
            "sessionCount": session_count + 1,
        }, merge=True)
    except Exception as e:
        print(f"Memory write error: {e}")

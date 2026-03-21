from fastapi import APIRouter, HTTPException
from app.models.schemas import MoodEntry
import os

router = APIRouter()

# In-memory fallback if Firebase is not configured
_mood_store: list = []


def get_firebase_db():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                return None

        return firestore.client()
    except Exception as e:
        print(f"Firebase init error: {e}")
        return None


@router.post("/mood")
async def log_mood(entry: MoodEntry):
    from datetime import datetime, timezone

    record = {
        "mood": entry.mood,
        "label": entry.label,
        "note": entry.note,
        "user_id": entry.user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    db = get_firebase_db()
    if db:
        try:
            db.collection("mood_entries").add(record)
            return {"success": True, "stored": "firebase"}
        except Exception as e:
            print(f"Firebase write error: {e}")

    # Fallback to in-memory
    _mood_store.append(record)
    return {"success": True, "stored": "memory"}


@router.get("/mood/{user_id}")
async def get_moods(user_id: str = "anonymous"):
    db = get_firebase_db()
    if db:
        try:
            docs = (
                db.collection("mood_entries")
                .where("user_id", "==", user_id)
                .order_by("timestamp", direction="DESCENDING")
                .limit(30)
                .stream()
            )
            entries = [doc.to_dict() for doc in docs]
            return {"entries": entries}
        except Exception as e:
            print(f"Firebase read error: {e}")

    # Fallback to in-memory
    user_entries = [e for e in _mood_store if e.get("user_id") == user_id]
    return {"entries": sorted(user_entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:30]}

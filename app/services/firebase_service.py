import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore

_app = None
_db = None
_disabled = False
_mem_db: Dict[str, Dict[str, Any]] = {}


def init_firebase():
    """Lazy init. If creds missing, disable persistence without crashing."""
    global _app, _db, _disabled
    if _disabled:
        return None
    if _app:
        return _db

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not cred_path or not os.path.exists(cred_path):
        _disabled = True
        print("FIREBASE_CREDENTIALS_PATH not set or missing — skipping Firestore persistence.")
        return None

    cred = credentials.Certificate(cred_path)
    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client()
    return _db


def save_message(user_id: str, chat_id: str, message: Dict[str, Any]):
    db = init_firebase()
    if db:
        ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
        # set chat metadata
        meta = {
            "updated_at": datetime.utcnow(),
            "last_message": message.get("content", "")[:120],
        }
        if message.get("role") == "user":
            meta.setdefault("title", message.get("content", "")[:40] or "Chat")
        ref.set(meta, merge=True)
        ref.collection("messages").add(message)
        return

    # Memory fallback
    user_store = _mem_db.setdefault(user_id, {"chats": {}})
    chat_store = user_store["chats"].setdefault(chat_id, {"messages": [], "title": "Chat", "updated_at": datetime.utcnow().isoformat()})
    chat_store["messages"].append(message)
    chat_store["updated_at"] = datetime.utcnow().isoformat()
    if message.get("role") == "user" and not chat_store.get("title"):
        chat_store["title"] = message.get("content", "")[:40] or "Chat"


def get_history(user_id: str) -> List[Dict[str, Any]]:
    db = init_firebase()
    if db:
        try:
            chats_ref = db.collection("users").document(user_id).collection("chats")
            snaps = chats_ref.order_by("updated_at", direction=firestore.Query.DESCENDING).stream()
            history = []
            for doc in snaps:
                data = doc.to_dict() or {}
                history.append({
                    "id": doc.id,
                    "title": data.get("title") or "Chat",
                    "updated_at": data.get("updated_at"),
                    "last_message": data.get("last_message", ""),
                })
            return history
        except Exception as exc:
            print(f"History fetch error: {exc}")

    # Memory fallback
    user_store = _mem_db.get(user_id, {}).get("chats", {})
    items = []
    for cid, chat in user_store.items():
        items.append({
            "id": cid,
            "title": chat.get("title") or "Chat",
            "updated_at": chat.get("updated_at"),
            "last_message": (chat.get("messages") or [])[-1].get("content", "") if chat.get("messages") else "",
        })
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items


def get_messages(user_id: str, chat_id: str) -> List[Dict[str, Any]]:
    db = init_firebase()
    if db:
        try:
            msgs_ref = db.collection("users").document(user_id).collection("chats").document(chat_id).collection("messages")
            snaps = msgs_ref.order_by("timestamp").stream()
            return [s.to_dict() for s in snaps]
        except Exception as exc:
            print(f"Messages fetch error: {exc}")

    # Memory fallback
    return _mem_db.get(user_id, {}).get("chats", {}).get(chat_id, {}).get("messages", [])

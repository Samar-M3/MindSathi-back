import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, Source
from app.services.ai.router import generate_ai_response
from app.services.search_service import web_search, build_context
from app.utils.decision_engine import needs_web_search
from app.services.firebase_service import save_message, get_history, get_messages

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    use_search = request.search_enabled and needs_web_search(query)
    search_results = []
    context = None

    if use_search:
        try:
            search_results = await web_search(query, num_results=5)
            context = build_context(search_results)
        except Exception as exc:
            print(f"Search error: {exc}")
            search_results = []
            context = None

    uid = request.uid or request.user_id or "anonymous"

    full_prompt = query
    if context:
        full_prompt = f"{query}\n\nContext:\n{context}"

    answer_text = await asyncio.to_thread(generate_ai_response, full_prompt)

    sources = []
    if search_results:
        sources = [Source(title=r["title"], link=r["link"]) for r in search_results]

    now_iso = datetime.utcnow().isoformat()
    user_msg = {"role": "user", "content": query, "timestamp": now_iso}
    bot_msg = {"role": "assistant", "content": answer_text, "timestamp": now_iso}

    try:
        # Always persist to storage (Firestore when available, in-memory fallback otherwise)
        save_message(uid or "anonymous", request.chat_id or "default", user_msg)
        save_message(uid or "anonymous", request.chat_id or "default", bot_msg)
    except Exception as exc:
        print(f"Firestore save error: {exc}")

    return ChatResponse(answer=answer_text, sources=sources, mood="neutral", confidence=0.5)


@router.post("/search")
async def search_endpoint(payload: dict):
    query = payload.get("q") or payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    results = await web_search(query)
    return {"results": results}


@router.get("/history")
async def history(uid: str, chat_id: str | None = None):
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")

    if chat_id:
        messages = await asyncio.to_thread(get_messages, uid, chat_id)
        return {"messages": messages, "chat_id": chat_id}

    chats = await asyncio.to_thread(get_history, uid)
    return {"chats": chats}

import asyncio
from fastapi import APIRouter, BackgroundTasks, Request
from app.services.memory_service import update_user_memory

router = APIRouter()


@router.post("/api/memory/update")
async def update_memory(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
        return {"status": "ok"}

    uid = payload.get("uid")
    messages = payload.get("messages") or []

    if not uid or uid == "anonymous":
        return {"status": "ok"}
    if len(messages) < 3:
        return {"status": "ok"}

    def _run_update():
        asyncio.run(update_user_memory(uid, messages))

    background_tasks.add_task(_run_update)
    return {"status": "ok"}

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes import chat
from app.routers.memory_router import router as memory_router
from app.routers.session_router import router as session_router

load_dotenv()

app = FastAPI(title="MindSathi Searchable Chat", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(memory_router)
app.include_router(session_router)


@app.get("/health")
async def health():
    return {"status": "ok"}

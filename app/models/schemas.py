from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatMessage] = []
    user_id: Optional[str] = "anonymous"
    uid: Optional[str] = "anonymous"
    chat_id: Optional[str] = "default"
    search_enabled: bool = True
    lang: Optional[str] = "en"


class Source(BaseModel):
    title: str
    link: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    mood: str = "neutral"
    confidence: float = 0.5


class AIResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    mood: str = "neutral"
    confidence: float = 0.5


class MoodEntry(BaseModel):
    mood: str
    label: str
    note: Optional[str] = ""
    user_id: str = "anonymous"

# models.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---- Core request objects ----

class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)
    created_at: Optional[str] = None  # ISO 8601 (e.g., "2025-09-07T10:05:00Z")


class IngestRequest(BaseModel):
    """
    Ingest raw turns; backend will windowize and insert into conv_windows
    with embedding=NULL (pending). test_group: 0=no RAG, 1=with RAG.
    """
    user_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    turns: List[Turn] = Field(..., min_items=1)
    test_group: Literal[0, 1] = 1


class AskRequest(BaseModel):
    """
    Ask a question for a given user (optionally a specific conversation).
    hybrid=True enables vector + FTS hybrid scoring.
    """
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    top_k: int = Field(6, ge=1, le=20)
    hybrid: bool = False
    test_group: Literal[0, 1] = 1  # pass-through for analytics logging


# ---- Retrieval / response objects ----

class Snippet(BaseModel):
    window_id: str
    conversation_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    snippets: List[Snippet]

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ColdStartRequest(BaseModel):
    age_bucket: str | None = None
    gender: str | None = None
    occupation: str | None = None
    favorite_genres: list[str] = Field(default_factory=list)
    top_k: int = 10


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    value: float | None = None
    context: dict[str, Any] | None = None

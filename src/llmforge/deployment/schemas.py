from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class GenerateResponse(BaseModel):
    completion: str


class HealthResponse(BaseModel):
    status: str
    model_path: str

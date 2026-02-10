from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
    id: int = Field(..., ge=1)
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    job_id: str
    file_name: str
    duration_seconds: float
    model_name: str
    diarization_enabled: bool
    detected_language: Optional[str] = None
    transcript: str
    segments: list[Segment] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str

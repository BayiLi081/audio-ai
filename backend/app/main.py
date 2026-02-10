from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import HealthResponse, TranscriptionResponse
from .services.audio import (
    convert_audio_to_wav,
    ensure_supported_extension,
    get_audio_duration_seconds,
    save_upload_file,
)
from .services.diarization import diarize_and_transcribe
from .services.transcription import transcribe_audio

app = FastAPI(
    title="Audio AI Web App API",
    description="Upload audio and receive transcription with optional speaker diarization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    settings.uploads_root.mkdir(parents=True, exist_ok=True)
    settings.processed_root.mkdir(parents=True, exist_ok=True)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="audio-ai-web-app")


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(settings.default_whisper_model),
    diarize: bool = Form(False),
) -> TranscriptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")

    try:
        ensure_supported_extension(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid4().hex
    upload_dir = settings.uploads_root / job_id
    output_dir = settings.processed_root / job_id

    try:
        source_path = await save_upload_file(file, upload_dir)
        wav_path = convert_audio_to_wav(source_path, output_dir)
        duration_seconds = get_audio_duration_seconds(wav_path)

        if diarize:
            diarization_config = settings.diarization_config_path
            if not diarization_config.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Diarization config not found at {diarization_config}",
                )

            diarized = diarize_and_transcribe(
                audio_wav_path=wav_path,
                output_dir=output_dir / "segments",
                diarization_config_path=diarization_config,
                model_name=model_name,
            )

            return TranscriptionResponse(
                job_id=job_id,
                file_name=file.filename,
                duration_seconds=duration_seconds,
                model_name=model_name,
                diarization_enabled=True,
                detected_language=None,
                transcript=diarized["transcript"],
                segments=diarized["segments"],
            )

        transcription = transcribe_audio(wav_path, model_name)

        return TranscriptionResponse(
            job_id=job_id,
            file_name=file.filename,
            duration_seconds=duration_seconds,
            model_name=model_name,
            diarization_enabled=False,
            detected_language=transcription.get("language"),
            transcript=transcription["text"],
            segments=[],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = settings.frontend_root / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail=f"Frontend not found at {index_path}")

    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=settings.frontend_root), name="static")

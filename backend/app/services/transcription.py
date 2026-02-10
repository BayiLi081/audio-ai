from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

_whisper_models: dict[str, Any] = {}
_whisper_lock = Lock()


def load_whisper_model(model_name: str) -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Install dependencies from backend/requirements.txt."
        ) from exc

    with _whisper_lock:
        if model_name not in _whisper_models:
            _whisper_models[model_name] = whisper.load_model(model_name)

    return _whisper_models[model_name]


def transcribe_audio(audio_wav_path: Path, model_name: str) -> dict[str, Any]:
    model = load_whisper_model(model_name)
    result = model.transcribe(str(audio_wav_path))

    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language"),
        "raw": result,
    }

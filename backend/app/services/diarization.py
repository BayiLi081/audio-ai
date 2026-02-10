from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from pydub import AudioSegment

from .transcription import load_whisper_model

_diarization_pipelines: dict[Path, Any] = {}
_diarization_lock = Lock()


def _load_pipeline_from_local_config(path_to_config: Path) -> Any:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install advanced dependencies to use diarization."
        ) from exc

    if not path_to_config.is_file():
        raise FileNotFoundError(f"Diarization config file not found: {path_to_config}")

    cwd = Path.cwd().resolve()
    change_to = path_to_config.parent.parent.resolve()

    os.chdir(change_to)
    try:
        pipeline = Pipeline.from_pretrained(path_to_config)
    finally:
        os.chdir(cwd)

    return pipeline


def load_diarization_pipeline(path_to_config: Path) -> Any:
    resolved_path = path_to_config.resolve()

    with _diarization_lock:
        if resolved_path not in _diarization_pipelines:
            _diarization_pipelines[resolved_path] = _load_pipeline_from_local_config(resolved_path)

    return _diarization_pipelines[resolved_path]


def diarize_and_transcribe(
    audio_wav_path: Path,
    output_dir: Path,
    diarization_config_path: Path,
    model_name: str,
) -> dict[str, Any]:
    pipeline = load_diarization_pipeline(diarization_config_path)
    whisper_model = load_whisper_model(model_name)

    diarization = pipeline(str(audio_wav_path))
    original_audio = AudioSegment.from_wav(str(audio_wav_path))

    output_dir.mkdir(parents=True, exist_ok=True)

    segments: list[dict[str, Any]] = []
    for index, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), start=1):
        raw_start = float(turn.start)
        raw_end = float(turn.end)
        if raw_end <= raw_start:
            continue

        audio_slice = original_audio[int(raw_start * 1000) : int(raw_end * 1000)]
        segment_path = output_dir / f"speaker_{speaker}_segment_{index}.wav"
        audio_slice.export(str(segment_path), format="wav")

        result = whisper_model.transcribe(str(segment_path))
        text = str(result.get("text", "")).strip()

        segments.append(
            {
                "id": index,
                "speaker": str(speaker),
                "start": round(raw_start, 2),
                "end": round(raw_end, 2),
                "text": text,
            }
        )

    segments = sorted(segments, key=lambda item: item["start"])
    transcript = " ".join(
        f"{segment['speaker']}: {segment['text']}" for segment in segments if segment["text"]
    ).strip()

    return {
        "segments": segments,
        "transcript": transcript,
    }

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile
from pydub import AudioSegment

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mp4",
    ".webm",
}


def ensure_supported_extension(file_name: str) -> None:
    extension = Path(file_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        readable = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{extension}'. Allowed: {readable}")


async def save_upload_file(upload_file: UploadFile, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(upload_file.filename or "uploaded_audio").name
    destination = destination_dir / file_name

    with destination.open("wb") as output:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    return destination


def convert_audio_to_wav(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.wav"

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(str(output_path), format="wav")

    return output_path


def get_audio_duration_seconds(audio_path: Path) -> float:
    audio = AudioSegment.from_file(str(audio_path))
    return round(len(audio) / 1000.0, 2)

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    backend_root: Path = project_root / "backend"
    data_root: Path = backend_root / "data"
    uploads_root: Path = data_root / "uploads"
    processed_root: Path = data_root / "processed"
    frontend_root: Path = project_root / "frontend"
    diarization_config_path: Path = project_root / "test" / "models" / "pyannote_diarization_config.yaml"
    default_whisper_model: str = "small.en"


settings = Settings()

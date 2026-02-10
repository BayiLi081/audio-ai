from __future__ import annotations

import csv
import json
from pathlib import Path

from ..schemas import TranscriptionResponse


def _dump_model(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    if hasattr(model, "dict"):
        return getattr(model, "dict")()
    raise TypeError(f"Unsupported model type for serialization: {type(model)!r}")


def save_transcription_outputs(output_dir: Path, result: TranscriptionResponse) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(result.transcript, encoding="utf-8")

    segments_csv_path = output_dir / "segments.csv"
    with segments_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "speaker", "start", "end", "text"],
        )
        writer.writeheader()
        for segment in result.segments:
            writer.writerow(_dump_model(segment))

    result_json_path = output_dir / "result.json"
    result_json_path.write_text(
        json.dumps(_dump_model(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "transcript": transcript_path,
        "segments_csv": segments_csv_path,
        "result_json": result_json_path,
    }

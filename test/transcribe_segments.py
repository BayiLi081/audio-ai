# transcribe_segments.py
"""
Cut a source WAV into speaker segments and transcribe each segment with Whisper.

Requirements:
- pydub (and FFmpeg installed on your system)
- openai-whisper (pip install -U openai-whisper)

Usage:
    from transcribe_segments import transcribe_speaker_segments
    df = transcribe_speaker_segments(
        speaker_durations=df,                    # DataFrame with columns: start, end, seg_unique_id, speaker
        audio_wav_path="/path/to/file.wav",      # Full path to the source WAV
        output_dir="test_audio",                 # Folder to save cut segments
        output_csv_path="/path/to/output.csv",   # Optional: save the updated DataFrame
        model_name="small.en"                    # Whisper model size/name
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pydub import AudioSegment
import whisper


def transcribe_speaker_segments(
    speaker_durations: pd.DataFrame,
    audio_wav_path: str | Path,
    output_dir: str | Path = "test_audio",
    output_csv_path: Optional[str | Path] = None,
    model_name: str = "small.en",
    model: Optional[whisper.Whisper] = None,
) -> pd.DataFrame:
    """
    Cut segments defined in `speaker_durations` from `audio_wav_path`, save each as WAV,
    and transcribe with Whisper. Returns the updated DataFrame (with a `text` column).
    
    Required DataFrame columns:
        - start (seconds, float or int)
        - end   (seconds, float or int)
        - seg_unique_id
        - speaker

    Args:
        speaker_durations: DataFrame of segment timings and IDs.
        audio_wav_path: Path to the source WAV file.
        output_dir: Folder to write per-segment WAV files.
        output_csv_path: If set, write the updated DataFrame to this CSV.
        model_name: Whisper model to load if `model` not provided.
        model: Optional preloaded Whisper model to reuse across calls.

    Returns:
        A copy of `speaker_durations` with a new `text` column and sorted by `start`.

    Notes:
        - Ensure FFmpeg is installed and on PATH for pydub.
        - For speed, consider passing a preloaded `model` across calls.
    """
    audio_wav_path = Path(audio_wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_wav_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_wav_path}")

    # Load once
    asr = model if model is not None else whisper.load_model(model_name)

    # Work on a copy to avoid mutating the caller's DataFrame
    df = speaker_durations.copy()
    if "text" not in df.columns:
        df["text"] = ""

    # Load the full audio
    original_audio = AudioSegment.from_wav(str(audio_wav_path))

    # Iterate rows
    for idx, row in df.iterrows():
        # Convert seconds to milliseconds for pydub slicing
        start_ms = int(float(row["start"]) * 1000)
        end_ms = int(float(row["end"]) * 1000)

        # Slice audio
        segment_audio = original_audio[start_ms:end_ms]

        # File naming
        seg_id = row["seg_unique_id"]
        speaker = row["speaker"]
        seg_filename = f"speaker_{speaker}_segment_{seg_id}.wav"
        seg_path = output_dir / seg_filename

        # Export segment
        segment_audio.export(str(seg_path), format="wav")

        # Transcribe
        result = asr.transcribe(str(seg_path))
        df.at[idx, "text"] = result.get("text", "")

    # Sort and save if requested
    df = df.sort_values(by=["start"]).reset_index(drop=True)

    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df


if __name__ == "__main__":
    # Minimal CLI for quick runs. Edit paths as needed.
    import argparse

    parser = argparse.ArgumentParser(description="Cut and transcribe speaker segments from a WAV.")
    parser.add_argument("--audio", required=True, help="Path to the source WAV file.")
    parser.add_argument("--segments_csv", required=True, help="CSV with columns: start,end,seg_unique_id,speaker")
    parser.add_argument("--out_dir", default="test_audio", help="Directory to write segment WAVs.")
    parser.add_argument("--out_csv", default=None, help="If set, write updated CSV to this path.")
    parser.add_argument("--model", default="small.en", help="Whisper model name (e.g. tiny, base, small.en, medium, large).")
    args = parser.parse_args()

    seg_df = pd.read_csv(args.segments_csv)
    updated = transcribe_speaker_segments(
        speaker_durations=seg_df,
        audio_wav_path=args.audio,
        output_dir=args.out_dir,
        output_csv_path=args.out_csv,
        model_name=args.model,
    )
    # Print a quick peek
    print(updated.head())

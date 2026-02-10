# Audio AI Web App

This project is now a full web app with a FastAPI backend and browser frontend.

## What it does

- Upload audio from the web UI
- Convert to 16kHz mono WAV
- Transcribe with Whisper
- Optionally run speaker diarization (using local pyannote config/models already in `test/models`)

## Project structure

- `backend/app/main.py`: API app and routes
- `backend/app/services/`: audio conversion, transcription, diarization services
- `frontend/index.html`: app shell
- `frontend/app.js`: client logic
- `frontend/styles.css`: responsive UI styling

## Prerequisites

- Python 3.10+
- FFmpeg installed and available in PATH (required by `pydub`)

## Install

```bash
pip install -r backend/requirements.txt
```

Optional diarization support:

```bash
pip install -r backend/requirements-diarization.txt
```

## Run

```bash
uvicorn backend.app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/` (frontend)
- `http://127.0.0.1:8000/docs` (Swagger API)

## API

- `GET /api/health`
- `POST /api/transcribe`
  - form fields:
    - `file` (audio file)
    - `model_name` (Whisper model, e.g. `small.en`)
    - `diarize` (`true`/`false`)

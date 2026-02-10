const form = document.getElementById("transcribe-form");
const statusElement = document.getElementById("status");
const submitButton = document.getElementById("submit-btn");
const resultBlock = document.getElementById("result-block");
const metaElement = document.getElementById("meta");
const transcriptElement = document.getElementById("transcript");
const segmentsElement = document.getElementById("segments");

function setStatus(message, level = "") {
  statusElement.textContent = message;
  statusElement.className = `status ${level}`.trim();
}

function formatTime(seconds) {
  return `${Number(seconds).toFixed(2)}s`;
}

function renderSegments(segments) {
  segmentsElement.innerHTML = "";

  if (!segments || segments.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "No segment breakdown available for this run.";
    segmentsElement.appendChild(empty);
    return;
  }

  segments.forEach((segment) => {
    const card = document.createElement("div");
    card.className = "segment";

    const meta = document.createElement("strong");
    meta.textContent = `${segment.speaker} | ${formatTime(segment.start)} - ${formatTime(segment.end)}`;

    const text = document.createElement("p");
    text.textContent = segment.text || "(No speech detected)";

    card.appendChild(meta);
    card.appendChild(text);
    segmentsElement.appendChild(card);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = document.getElementById("audio-file");
  const modelInput = document.getElementById("model-name");
  const diarizeInput = document.getElementById("diarize");

  if (!fileInput.files || fileInput.files.length === 0) {
    setStatus("Please choose an audio file first.", "error");
    return;
  }

  const payload = new FormData();
  payload.append("file", fileInput.files[0]);
  payload.append("model_name", modelInput.value);
  payload.append("diarize", diarizeInput.checked ? "true" : "false");

  submitButton.disabled = true;
  setStatus("Processing audio... this can take a while for larger files.");

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: payload,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Transcription failed.");
    }

    resultBlock.classList.remove("hidden");
    metaElement.textContent = `Job ${data.job_id.slice(0, 8)} | ${data.file_name} | ${formatTime(
      data.duration_seconds,
    )} | model: ${data.model_name}`;
    transcriptElement.textContent = data.transcript || "(No transcript generated)";
    renderSegments(data.segments);

    setStatus("Transcription complete.", "ok");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", "error");
  } finally {
    submitButton.disabled = false;
  }
});

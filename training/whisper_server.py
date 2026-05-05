#!/usr/bin/env python3
"""Tiny FastAPI server for mlx-whisper. Local-only, no auth.

Endpoints:
  GET  /health          -> {"ok": true, "model": "..."}
  POST /v1/transcribe   -> multipart/form-data with field 'audio' (any audio file)
                            optional 'language' field (BCP-47 like 'en', 'es', etc)
                            -> {"text": "...", "language": "...", "latency_ms": int}

Run:
  source venv/bin/activate
  python3 whisper_server.py  # listens on 0.0.0.0:8002
"""
import os
import time
import tempfile
import logging

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlx_whisper

MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

app = FastAPI(title="prism-whisper", version="1.0")

# Allow prism-aac (localhost:3000) to call us from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whisper")


@app.on_event("startup")
async def _warm():
    """Warm the model so first user request isn't 4s slower."""
    log.info(f"Warming {MODEL}...")
    # mlx-whisper auto-downloads + caches; calling transcribe on a 0.1s
    # silence file is the cheapest way to load the weights into MLX cache.
    import wave, struct
    p = tempfile.mktemp(suffix=".wav")
    with wave.open(p, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 1600, *([0] * 1600)))  # 0.1s silence
    try:
        mlx_whisper.transcribe(p, path_or_hf_repo=MODEL)
    except Exception as e:
        log.warning(f"warmup failed: {e}")
    finally:
        try:
            os.unlink(p)
        except Exception:
            pass
    log.info("ready")


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL}


@app.post("/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
):
    if not audio:
        raise HTTPException(400, "missing audio file")
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    tmp = tempfile.mktemp(suffix=suffix)
    try:
        with open(tmp, "wb") as f:
            f.write(await audio.read())
        t0 = time.time()
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = mlx_whisper.transcribe(tmp, path_or_hf_repo=MODEL, **kwargs)
        dt_ms = int((time.time() - t0) * 1000)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language or "?"),
            "latency_ms": dt_ms,
        }
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("whisper_server:app", host="0.0.0.0", port=8002, log_level="info")

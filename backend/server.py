"""NeuTTS FastAPI backend — runs on HuggingFace Spaces."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from neutts import NeuTTS

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY     = os.environ.get("NEUTTS_API_KEY", "")
BACKBONE    = os.environ.get("NEUTTS_BACKBONE", "neuphonic/neutts-nano-q8-gguf")
DEVICE      = os.environ.get("NEUTTS_DEVICE",   "cpu")
CODEC       = os.environ.get("NEUTTS_CODEC",    "neuphonic/neucodec-onnx-decoder")
SAMPLE_RATE = 24_000

# ─── Model loading (at startup) ───────────────────────────────────────────────

print(f"[backend] Loading NeuTTS: backbone={BACKBONE}  device={DEVICE}  codec={CODEC}", flush=True)
_tts: NeuTTS | None = None
try:
    _tts = NeuTTS(
        backbone_repo=BACKBONE,
        backbone_device=DEVICE,
        codec_repo=CODEC,
        codec_device="cpu",
    )
    print("[backend] Model loaded OK", flush=True)
except Exception as exc:
    print(f"[backend] WARNING: model load failed: {exc}", file=sys.stderr, flush=True)

_whisper_model = None
_whisper_model_name = ""

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="NeuTTS backend", version="1.0")


def _check_key(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health(x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    return {
        "status": "ok",
        "model_loaded": _tts is not None,
        "backbone": BACKBONE,
        "device": DEVICE,
        "codec": CODEC,
    }


@app.post("/generate")
async def generate(
    text:        str        = Form(...),
    ref_text:    str        = Form(""),
    temperature: float      = Form(1.0),
    top_k:       int        = Form(50),
    ref_audio:   UploadFile = File(...),
    x_api_key:   str | None = Header(default=None),
):
    _check_key(x_api_key)
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded on backend")

    suffix = Path(ref_audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    try:
        ref_codes, _ = _tts.encode_reference(tmp_path)
        wav = _tts.infer(
            text.strip(),
            ref_codes,
            ref_text.strip() or " ",
            temperature=float(temperature),
            top_k=int(top_k),
        )
        buf = io.BytesIO()
        sf.write(buf, wav.astype(np.float32), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as exc:
        print(f"[backend] /generate error:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe(
    audio:     UploadFile = File(...),
    model_id:  str        = Form("base"),
    x_api_key: str | None = Header(default=None),
):
    global _whisper_model, _whisper_model_name
    _check_key(x_api_key)

    try:
        import whisper as _w
    except ImportError:
        raise HTTPException(status_code=503, detail="openai-whisper not installed on backend")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        if _whisper_model is None or _whisper_model_name != model_id:
            print(f"[backend] loading Whisper '{model_id}'...", flush=True)
            _whisper_model = _w.load_model(model_id)
            _whisper_model_name = model_id
        result = _whisper_model.transcribe(tmp_path)
        return {"text": result["text"].strip()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

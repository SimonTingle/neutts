"""NeuTTS local UI — Gradio frontend for on-device voice synthesis."""

from __future__ import annotations

import subprocess
import sys
import time
import traceback
import warnings
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

from neutts import NeuTTS

# Suppress repetitive third-party deprecation noise that isn't actionable.
# Our own _log() calls replace these with clear, structured output.
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Redirects are currently not supported")

# ─── Terminal logging ─────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO") -> None:
    tag = {"INFO": "[neutts]", "WARN": "[neutts WARN]", "ERROR": "[neutts ERROR]"}[level]
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"{tag} {msg}", flush=True, file=stream)


# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24_000
MAX_TEXT_CHARS = 500
MIN_REF_SECS = 3.0
MAX_REF_SECS = 30.0

GGUF_MODELS = [
    "neuphonic/neutts-nano-q8-gguf",
    "neuphonic/neutts-nano-q4-gguf",
    "neuphonic/neutts-air-q8-gguf",
    "neuphonic/neutts-air-q4-gguf",
    "neuphonic/neutts-nano-german-q8-gguf",
    "neuphonic/neutts-nano-french-q8-gguf",
    "neuphonic/neutts-nano-spanish-q8-gguf",
    "neuphonic/neutts-nano-german-q4-gguf",
    "neuphonic/neutts-nano-french-q4-gguf",
    "neuphonic/neutts-nano-spanish-q4-gguf",
]
TORCH_MODELS = [
    "neuphonic/neutts-nano",
    "neuphonic/neutts-air",
]
ALL_MODELS = GGUF_MODELS + TORCH_MODELS

_DEVICES = ["auto", "cpu"]
if torch.backends.mps.is_available():
    _DEVICES.insert(1, "metal")
if torch.cuda.is_available():
    _DEVICES.insert(1, "cuda")
DEVICES = _DEVICES

CODEC_REPOS: dict[str, str] = {
    "ONNX decoder  (fastest · CPU only)":         "neuphonic/neucodec-onnx-decoder",
    "ONNX int8 decoder  (smallest · CPU only)":   "neuphonic/neucodec-onnx-decoder-int8",
    "NeuCodec  (GPU-capable)":                    "neuphonic/neucodec",
    "DistillNeuCodec  (lightweight · GPU-capable)": "neuphonic/distill-neucodec",
}
ONNX_CODECS = frozenset({
    "neuphonic/neucodec-onnx-decoder",
    "neuphonic/neucodec-onnx-decoder-int8",
})

# Built-in sample speakers shipped with the repo
_SAMPLES_DIR = Path(__file__).parent / "samples"
_SAMPLE_SPEAKERS: dict[str, tuple[str, str]] = {}  # label → (wav_path, txt_path)
for _wav in sorted(_SAMPLES_DIR.glob("*.wav")):
    _txt = _wav.with_suffix(".txt")
    if _txt.exists():
        _SAMPLE_SPEAKERS[_wav.stem.capitalize()] = (str(_wav), _txt.read_text().strip())
SAMPLE_CHOICES = ["— custom upload —"] + list(_SAMPLE_SPEAKERS)

# Formats natively readable by soundfile (no ffmpeg needed)
_SOUNDFILE_FORMATS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".snd"}

# ─── Singleton state ──────────────────────────────────────────────────────────

_tts: NeuTTS | None = None
_loaded_cfg: dict = {}
_ref_cache: dict[str, object] = {}   # audio file path → encoded ref codes
_fallback_encoder = None              # NeuCodec loaded lazily for ONNX-only setups
_converted_paths: dict[str, str] = {}  # original path → converted WAV path
_whisper_model = None                 # cached Whisper model for auto-transcription
_whisper_model_name: str = ""         # which model is currently loaded

WHISPER_MODELS = [
    ("tiny   — 39 MB  · fastest, rough",      "tiny"),
    ("tiny.en — 39 MB  · English-only, faster", "tiny.en"),
    ("base   — 74 MB  · default",             "base"),
    ("base.en — 74 MB  · English-only",         "base.en"),
    ("small  — 244 MB · better accuracy",     "small"),
    ("small.en — 244 MB · English-only",       "small.en"),
    ("medium — 769 MB · strong accuracy",     "medium"),
    ("medium.en — 769 MB · English-only",      "medium.en"),
    ("large-v3 — 1.5 GB · best (slow on CPU)", "large-v3"),
]
WHISPER_MODEL_CHOICES = [label for label, _ in WHISPER_MODELS]
WHISPER_MODEL_DEFAULT = "base   — 74 MB  · default"
_WHISPER_LABEL_TO_ID = {label: mid for label, mid in WHISPER_MODELS}

# ─── Presets ──────────────────────────────────────────────────────────────────

BUILT_IN_PRESETS = {
    "🍎 Apple Silicon Metal (fast, streaming)": {
        "backbone": "neuphonic/neutts-nano-q8-gguf",
        "device": "metal",
        "codec": "ONNX decoder  (fastest · CPU only)",
        "temperature": 1.0,
        "top_k": 50,
        "streaming": True,
    },
    "⚡ Fast CPU (low latency)": {
        "backbone": "neuphonic/neutts-nano-q8-gguf",
        "device": "cpu",
        "codec": "ONNX decoder  (fastest · CPU only)",
        "temperature": 0.1,
        "top_k": 0,
        "streaming": True,
    },
    "🎨 Creative (varied output)": {
        "backbone": "neuphonic/neutts-nano",
        "device": "auto",
        "codec": "NeuCodec  (GPU-capable)",
        "temperature": 1.8,
        "top_k": 50,
        "streaming": False,
    },
    "🎯 Accurate (conservative)": {
        "backbone": "neuphonic/neutts-nano",
        "device": "auto",
        "codec": "NeuCodec  (GPU-capable)",
        "temperature": 0.3,
        "top_k": 20,
        "streaming": False,
    },
    "🖥️ GPU CUDA (PyTorch)": {
        "backbone": "neuphonic/neutts-nano",
        "device": "cuda",
        "codec": "NeuCodec  (GPU-capable)",
        "temperature": 1.0,
        "top_k": 50,
        "streaming": False,
    },
    "💾 High Quality (slower)": {
        "backbone": "neuphonic/neutts-nano",
        "device": "auto",
        "codec": "NeuCodec  (GPU-capable)",
        "temperature": 0.7,
        "top_k": 100,
        "streaming": False,
    },
}

_saved_presets: dict[str, dict] = {}  # user-saved custom presets


# ─── Audio format conversion ──────────────────────────────────────────────────

def _convert_audio_to_wav(path: str) -> str:
    """Convert non-WAV audio to 16 kHz mono WAV using ffmpeg. Cached by path."""
    if path in _converted_paths:
        cached = _converted_paths[path]
        _log(f"  audio: using cached conversion → {Path(cached).name}")
        return cached

    suffix = Path(path).suffix.lower()
    size_kb = Path(path).stat().st_size / 1024 if Path(path).exists() else 0
    _log(f"  audio: received {Path(path).name}  ({suffix[1:].upper()}, {size_kb:.0f} KB)")

    if suffix in _SOUNDFILE_FORMATS:
        _log(f"  audio: format {suffix[1:].upper()} is natively supported — no conversion needed")
        _converted_paths[path] = path
        return path

    out_path = path + "_neutts.wav"
    _log(f"  audio: {suffix[1:].upper()} requires conversion — running ffmpeg → WAV 16 kHz mono...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            out_kb = Path(out_path).stat().st_size / 1024
            _log(f"  audio: conversion OK → {Path(out_path).name}  ({out_kb:.0f} KB)")
            _converted_paths[path] = out_path
            return out_path
        else:
            _log(f"  audio: ffmpeg conversion FAILED (exit {result.returncode})", "ERROR")
            for line in result.stderr.strip().splitlines():
                _log(f"    ffmpeg: {line}", "ERROR")
            if "Library not loaded" in result.stderr or "dyld" in result.stderr:
                _log("  audio: broken ffmpeg install — fix with: brew reinstall x265 && brew reinstall ffmpeg", "WARN")
            _log("  audio: falling back to original file — librosa will attempt to load it")
    except FileNotFoundError:
        _log("  audio: ffmpeg not found — install with: brew install ffmpeg", "WARN")
        _log("  audio: falling back to librosa audioread (file may still load)", "WARN")

    _log(f"  audio: using original file as fallback — librosa will attempt to load {Path(path).name}")
    _converted_paths[path] = path  # cache original so we don't retry on every keystroke
    return path


# ─── Model management ─────────────────────────────────────────────────────────

def load_model(backbone: str, device: str, codec_label: str) -> str:
    global _tts, _loaded_cfg, _ref_cache, _fallback_encoder

    codec_repo = CODEC_REPOS[codec_label]
    # ONNX decoders run CPU-only; full codecs can share the backbone device.
    codec_device = "cpu" if codec_repo in ONNX_CODECS else device
    cfg = {"backbone": backbone, "device": device, "codec": codec_repo}

    _log(f"load_model: backbone={backbone}  device={device}  codec={codec_repo}  codec_device={codec_device}")

    if _tts is not None and cfg == _loaded_cfg:
        _log("load_model: config unchanged — reusing loaded model")
        return "✓ Already loaded — settings unchanged."

    _log("load_model: clearing previous model and caches")
    _tts = None
    _loaded_cfg = {}
    _ref_cache.clear()
    _fallback_encoder = None

    try:
        _log("load_model: instantiating NeuTTS...")
        _tts = NeuTTS(
            backbone_repo=backbone,
            backbone_device=device,
            codec_repo=codec_repo,
            codec_device=codec_device,
        )
        _loaded_cfg = cfg
        stream_note = "streaming ✓" if _tts._is_quantized_model else "streaming ✗ (GGUF only)"
        _log(f"load_model: OK  {stream_note}")
        return f"✓ {backbone}\nDevice: {device}  ·  {stream_note}"
    except Exception:
        tb = traceback.format_exc()
        _log(f"load_model: FAILED\n{tb}", "ERROR")
        return f"✗ Load failed:\n{tb}"


# ─── Reference encoding ───────────────────────────────────────────────────────

def _encode_reference(audio_path: str) -> tuple[object, str | None]:
    """Return (ref_codes, optional_warning). Results are cached by path."""
    global _fallback_encoder

    if audio_path in _ref_cache:
        _log(f"encode_reference: cache hit for {Path(audio_path).name}")
        return _ref_cache[audio_path], None

    _log(f"encode_reference: encoding {Path(audio_path).name} ...")
    warning: str | None = None
    try:
        codes = _tts.encode_reference(audio_path)
        _log(f"encode_reference: OK  shape={getattr(codes, 'shape', type(codes).__name__)}")
    except (AttributeError, RuntimeError) as exc:
        # ONNX decoders are decode-only — load a separate NeuCodec encoder.
        _log(f"encode_reference: codec lacks encoder ({type(exc).__name__}: {exc})", "WARN")
        if _fallback_encoder is None:
            _log("encode_reference: loading fallback NeuCodec encoder (one-time, ~1 GB download)...")
            from neucodec import NeuCodec  # noqa: PLC0415
            _fallback_encoder = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to("cpu")
            _log("encode_reference: fallback encoder loaded on cpu")
            warning = "Loaded separate NeuCodec encoder for ONNX-decoder compatibility (one-time)."
        else:
            _log("encode_reference: using cached fallback encoder")
        _log(f"encode_reference: loading audio at 16 kHz for fallback encoder...")
        wav, _ = librosa.load(audio_path, sr=16_000, mono=True)
        _log(f"encode_reference: audio loaded  {len(wav)} samples @ 16 kHz")
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            codes = _fallback_encoder.encode_code(audio_or_path=wav_t).squeeze(0).squeeze(0)
        _log(f"encode_reference: fallback encode OK  shape={codes.shape}")

    _ref_cache[audio_path] = codes
    return codes, warning


# ─── Input validation ─────────────────────────────────────────────────────────

def _check_ref_audio(path: str | None) -> tuple[bool, str]:
    if not path:
        _log("check_ref_audio: path is None/empty → no file uploaded")
        return False, "No file uploaded."
    _log(f"check_ref_audio: loading {Path(path).name} to verify duration...")
    try:
        wav, sr = librosa.load(path, sr=None, mono=True)
    except Exception as exc:
        _log(f"check_ref_audio: librosa.load FAILED: {exc}", "ERROR")
        return False, f"Cannot read file: {exc}"
    dur = len(wav) / sr
    _log(f"check_ref_audio: {dur:.2f}s @ {sr} Hz  ({len(wav)} samples)")
    if dur < MIN_REF_SECS:
        _log(f"check_ref_audio: REJECTED — too short ({dur:.1f}s < {MIN_REF_SECS:.0f}s)", "WARN")
        return False, f"Too short: {dur:.1f}s  (min {MIN_REF_SECS:.0f}s)"
    if dur > MAX_REF_SECS:
        _log(f"check_ref_audio: REJECTED — too long ({dur:.1f}s > {MAX_REF_SECS:.0f}s)", "WARN")
        return False, f"Too long: {dur:.1f}s  (max {MAX_REF_SECS:.0f}s)"
    _log(f"check_ref_audio: OK")
    return True, f"✓ {dur:.1f}s"


def _check_text(text: str | None) -> tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "Empty."
    if len(t) > MAX_TEXT_CHARS:
        _log(f"check_text: REJECTED — {len(t)} chars exceeds {MAX_TEXT_CHARS} limit", "WARN")
        return False, f"{len(t)} / {MAX_TEXT_CHARS} — too long."
    return True, f"{len(t)} / {MAX_TEXT_CHARS}"


def on_ref_audio_change(path):
    _log("─" * 50)
    _log(f"on_ref_audio_change: path={path!r}")
    if path:
        path = _convert_audio_to_wav(path)
    ok, msg = _check_ref_audio(path)
    icon = "✓" if ok else "✗"
    return f"{icon} {msg}"


def on_text_change(text):
    _, msg = _check_text(text)
    return msg


def on_ref_text_change(text):
    n = len(text or "")
    _log(f"ref_text change: {n} chars — {repr((text or '')[:60])}")
    phones_str = ""
    if text and text.strip():
        if _tts is not None:
            try:
                phones_str = _tts._to_phones(text.strip())
                n_tokens = len(phones_str.split())
                _log(f"  phonemes ({n_tokens} tokens): {repr(phones_str[:100])}")
            except Exception as e:
                phones_str = f"(phonemisation error: {e})"
        else:
            phones_str = "(load a model to preview phonemes)"
    return f"{n} chars", phones_str


def transcribe_ref_audio(audio_path: str | None, model_label: str = WHISPER_MODEL_DEFAULT) -> str:
    """Auto-transcribe reference audio using Whisper and return the text."""
    global _whisper_model, _whisper_model_name
    if not audio_path:
        _log("transcribe: no audio path", "WARN")
        return ""
    try:
        import whisper as _whisper_pkg
    except ImportError:
        _log("transcribe: openai-whisper not installed", "WARN")
        return "⚠ openai-whisper not installed — run:  pip install openai-whisper"

    audio_path = _convert_audio_to_wav(audio_path)
    model_id = _WHISPER_LABEL_TO_ID.get(model_label, "base")

    if _whisper_model is None or _whisper_model_name != model_id:
        size_hint = {
            "tiny": "~39 MB", "tiny.en": "~39 MB",
            "base": "~74 MB", "base.en": "~74 MB",
            "small": "~244 MB", "small.en": "~244 MB",
            "medium": "~769 MB", "medium.en": "~769 MB",
            "large-v3": "~1.5 GB",
        }.get(model_id, "")
        _log(f"transcribe: loading Whisper '{model_id}' model ({size_hint}, one-time download)...")
        try:
            _whisper_model = _whisper_pkg.load_model(model_id)
            _whisper_model_name = model_id
            _log(f"transcribe: Whisper '{model_id}' loaded")
        except Exception as e:
            _log(f"transcribe: model load failed: {e}", "ERROR")
            return f"⚠ Whisper load failed: {e}"

    _log(f"transcribe: transcribing {Path(audio_path).name} with '{model_id}'...")
    try:
        result = _whisper_model.transcribe(audio_path)
        text = result["text"].strip()
        _log(f"transcribe: result = {repr(text)}")
        return text
    except Exception as e:
        _log(f"transcribe: failed: {e}", "ERROR")
        return f"⚠ Transcription failed: {e}"


def validate_output(audio_tuple, original_text):
    """Validate synthesized output by transcribing it and comparing to original text."""
    if audio_tuple is None or original_text is None or original_text.strip() == "":
        return "⚠ Need both synthesized audio and original transcript to validate."

    if _whisper_model is None:
        return "⚠ Whisper model not loaded — transcribe reference audio first."

    try:
        import soundfile as sf
        if isinstance(audio_tuple, tuple):
            sample_rate, audio_data = audio_tuple
        else:
            audio_data = audio_tuple
            sample_rate = 24000

        temp_wav = "/tmp/neutts_validate.wav"
        sf.write(temp_wav, audio_data, sample_rate)
        _log(f"validate: transcribing synthesized output...", "INFO")

        result = _whisper_model.transcribe(temp_wav)
        transcribed = result["text"].strip().lower()
        original = original_text.strip().lower()

        match_ratio = SequenceMatcher(None, original, transcribed).ratio()
        accuracy_pct = int(match_ratio * 100)

        report = f"**Output validation: {accuracy_pct}% match**\n\n"
        report += f"**Original:** {original}\n\n"
        report += f"**Transcribed:** {transcribed}\n\n"

        if accuracy_pct >= 90:
            report += "✓ Excellent — output is clear and matches input."
        elif accuracy_pct >= 75:
            report += "⚠ Good — minor differences detected."
        elif accuracy_pct >= 50:
            report += "✗ Poor — significant differences, check for garbling."
        else:
            report += "✗ Failed — output is garbled or unrecognizable."

        _log(f"validate: {accuracy_pct}% match", "INFO")
        return report
    except Exception as e:
        _log(f"validate: failed: {e}", "ERROR")
        return f"⚠ Validation failed: {e}"


def load_preset(preset_name):
    """Load a preset and return updates for all controls + success indicator."""
    all_presets = {**BUILT_IN_PRESETS, **_saved_presets}
    if preset_name not in all_presets:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "⚠ Preset not found")

    p = all_presets[preset_name]
    _log(f"preset: loading '{preset_name}'", "INFO")

    # Return updates for: backbone, device, codec, temperature, top_k, streaming, status
    return (
        gr.update(value=p.get("backbone", "neuphonic/neutts-nano")),
        gr.update(value=p.get("device", "auto")),
        gr.update(value=p.get("codec", "NeuCodec  (GPU-capable)")),
        gr.update(value=p.get("temperature", 1.0)),
        gr.update(value=p.get("top_k", 50)),
        gr.update(value=p.get("streaming", False)),
        "✓ Preset loaded",
    )


def save_preset(preset_name, backbone, device, codec, temperature, top_k, streaming):
    """Save current settings as a custom preset."""
    if not preset_name or preset_name.strip() == "":
        return "⚠ Preset name cannot be empty"
    if preset_name in BUILT_IN_PRESETS:
        return "⚠ Cannot overwrite built-in presets"

    _saved_presets[preset_name] = {
        "backbone": backbone,
        "device": device,
        "codec": codec,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "streaming": bool(streaming),
    }
    _log(f"preset: saved '{preset_name}'", "INFO")
    return f"✓ Preset '{preset_name}' saved"


def clear_model_cache():
    """Delete all cached models (HuggingFace backbone, codec, Whisper) to free disk space."""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "whisper",
    ]

    total_freed_kb = 0
    deleted_items = []

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        try:
            for item in cache_dir.iterdir():
                try:
                    if item.is_dir():
                        size_kb = sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        ) // 1024
                        shutil.rmtree(item)
                        deleted_items.append(f"{item.name} ({size_kb} KB)")
                        total_freed_kb += size_kb
                    elif item.is_file():
                        size_kb = item.stat().st_size // 1024
                        item.unlink()
                        deleted_items.append(f"{item.name} ({size_kb} KB)")
                        total_freed_kb += size_kb
                except Exception as e:
                    _log(f"clear_cache: error deleting {item.name}: {e}", "WARN")

        except Exception as e:
            _log(f"clear_cache: error accessing {cache_dir}: {e}", "WARN")

    total_freed_mb = total_freed_kb / 1024
    if deleted_items:
        report = f"✓ Freed **{total_freed_mb:.1f} MB**\n\n"
        report += "**Deleted:**\n" + "\n".join(f"• {item}" for item in deleted_items[:20])
        if len(deleted_items) > 20:
            report += f"\n• ... and {len(deleted_items) - 20} more items"
        _log(f"clear_cache: freed {total_freed_mb:.1f} MB", "INFO")
        return report
    else:
        return "ℹ No cached models found — disk cache already clean."


def on_sample_select(choice):
    """Fill reference audio + transcript from a built-in sample speaker."""
    if choice == "— custom upload —" or choice not in _SAMPLE_SPEAKERS:
        return gr.update(), gr.update()
    wav_path, transcript = _SAMPLE_SPEAKERS[choice]
    _log(f"on_sample_select: {choice} → {Path(wav_path).name}")
    return gr.update(value=wav_path), gr.update(value=transcript)


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(text, ref_audio, ref_text, streaming, temperature, top_k):
    _log("=" * 60)
    _log("generate: called")
    _log(f"  text        : {repr(text)[:80]}")
    _log(f"  ref_audio   : {ref_audio!r}")
    _log(f"  ref_text    : {repr(ref_text)[:60]}")
    _log(f"  streaming   : {streaming}  temperature: {temperature}  top_k: {top_k}")

    if _tts is None:
        _log("generate: no model loaded — aborting", "ERROR")
        yield None, "✗ No model loaded — click **Load Model** first."
        return

    text = (text or "").strip()
    ref_text = (ref_text or "").strip()

    # Convert audio format if needed before validation
    if ref_audio:
        _log("generate: checking/converting reference audio format...")
        ref_audio = _convert_audio_to_wav(ref_audio)

    errors: list[str] = []
    ok_t, msg_t = _check_text(text)
    if not ok_t:
        errors.append(f"Input text: {msg_t}")
    ok_r, msg_r = _check_ref_audio(ref_audio)
    if not ok_r:
        errors.append(f"Reference audio: {msg_r}")
    if not ref_text:
        # Transcript is optional — model still generates without it, voice
        # cloning accuracy may be slightly reduced.
        _log("generate: ref_text empty — proceeding without reference transcript", "WARN")
        ref_text = " "

    if errors:
        _log(f"generate: validation failed — {errors}", "ERROR")
        yield None, "✗ " + "  |  ".join(errors)
        return

    _log("generate: validation passed")

    use_stream = streaming and _tts._is_quantized_model
    if streaming and not _tts._is_quantized_model:
        _log("generate: streaming requested but backbone is not GGUF — falling back to non-streaming", "WARN")

    _log(f"generate: encoding reference audio (use_stream={use_stream})...")
    try:
        ref_codes, enc_warn = _encode_reference(ref_audio)
        _log(f"generate: reference encoded  enc_warn={enc_warn!r}")
    except Exception:
        tb = traceback.format_exc()
        _log(f"generate: reference encoding FAILED\n{tb}", "ERROR")
        yield None, f"✗ Reference encoding failed:\n```\n{tb}\n```"
        return

    note = f"\n_{enc_warn}_" if enc_warn else ""
    t0 = time.perf_counter()
    _log(f"generate: starting inference  text_len={len(text)}")

    try:
        if use_stream:
            _log("generate: streaming inference...")
            chunks: list[np.ndarray] = []
            for i, chunk in enumerate(_tts.infer_stream(
                text, ref_codes, ref_text,
                temperature=float(temperature),
                top_k=int(top_k),
            )):
                chunks.append(chunk)
                audio = np.concatenate(chunks).astype(np.float32)
                elapsed = time.perf_counter() - t0
                audio_s = len(audio) / SAMPLE_RATE
                rtf = elapsed / audio_s if audio_s > 0 else 0.0
                _log(f"  chunk {i+1}: {len(chunk)} samples  total={audio_s:.2f}s  RTF={rtf:.3f}")
                stats = f"⏱ {elapsed:.2f}s elapsed  ·  {audio_s:.2f}s audio  ·  RTF {rtf:.2f}{note}"
                yield (SAMPLE_RATE, audio), stats
            elapsed = time.perf_counter() - t0
            _log(f"generate: streaming done  {len(chunks)} chunks  elapsed={elapsed:.2f}s")
        else:
            _log("generate: non-streaming inference...")
            wav = _tts.infer(
                text, ref_codes, ref_text,
                temperature=float(temperature),
                top_k=int(top_k),
            )
            elapsed = time.perf_counter() - t0
            audio_s = len(wav) / SAMPLE_RATE
            rtf = elapsed / audio_s if audio_s > 0 else 0.0
            _log(f"generate: done  {len(wav)} samples  {audio_s:.2f}s audio  elapsed={elapsed:.2f}s  RTF={rtf:.3f}")
            stats = f"✓ {elapsed:.2f}s  ·  {audio_s:.2f}s audio  ·  RTF {rtf:.2f}{note}"
            yield (SAMPLE_RATE, wav.astype(np.float32)), stats

    except ValueError as exc:
        tb = traceback.format_exc()
        _log(f"generate: inference FAILED\n{tb}", "ERROR")
        if "No valid speech tokens" in str(exc):
            msg = (
                "✗ Model produced no speech tokens.\n\n"
                "**Likely fix:** tick the **Stream output** checkbox — "
                "GGUF models on Apple Metal work reliably in streaming mode. "
                "Non-streaming may produce empty output on MPS."
            )
            yield None, msg
        else:
            yield None, f"✗ Generation failed:\n```\n{tb}\n```"
    except Exception:
        tb = traceback.format_exc()
        _log(f"generate: inference FAILED\n{tb}", "ERROR")
        yield None, f"✗ Generation failed:\n```\n{tb}\n```"


# ─── UI layout ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    default_backbone = (
        "neuphonic/neutts-nano-q8-gguf" if (has_mps or has_cuda) else "neuphonic/neutts-nano"
    )
    default_device = "metal" if has_mps else ("cuda" if has_cuda else "cpu")
    default_codec = (
        "ONNX decoder  (fastest · CPU only)"
        if default_backbone.endswith("gguf")
        else "NeuCodec  (GPU-capable)"
    )

    with gr.Blocks(title="NeuTTS") as demo:
        gr.Markdown("# NeuTTS — Local Voice Synthesis")
        gr.Markdown(
            "On-device TTS with instant voice cloning.  "
            "**Text to synthesise** → the new words you want spoken.  "
            "**Reference audio** → a 3–30s clip of the target voice.  "
            "**Reference transcript** → type exactly what is said *in that clip* (not the new text)."
        )

        with gr.Row():

            # ── Left: model settings ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=270):
                gr.Markdown("### Presets")
                preset_choices = list(BUILT_IN_PRESETS.keys())
                preset_dd = gr.Dropdown(
                    choices=preset_choices,
                    label="Load preset",
                    interactive=True,
                )
                with gr.Row():
                    load_preset_btn = gr.Button("Load", size="sm", scale=1)
                    preset_status = gr.Markdown("", scale=2)

                gr.Markdown("### Model")
                backbone_dd = gr.Dropdown(ALL_MODELS, value=default_backbone, label="Backbone")
                device_dd   = gr.Dropdown(DEVICES, value=default_device, label="Device")
                codec_dd    = gr.Dropdown(list(CODEC_REPOS), value=default_codec, label="Codec")
                load_btn    = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Status", value="No model loaded.",
                    interactive=False, lines=3,
                )

                gr.Markdown("### Sampling")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                top_k       = gr.Slider(0, 100, value=50, step=1,  label="Top-K  (0 = disabled)")

                gr.Markdown("### Save Preset")
                preset_name = gr.Textbox(
                    label="New preset name",
                    placeholder="e.g. My custom voice",
                    max_lines=1,
                )
                save_preset_btn = gr.Button("Save preset", size="sm", variant="secondary")
                save_status = gr.Markdown("")

                gr.Markdown("### Disk Cleanup")
                clear_btn = gr.Button("🗑️ Clear all cached models", size="sm", variant="stop")
                clear_status = gr.Markdown("")

            # ── Right: I/O ───────────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Input")

                input_text = gr.Textbox(
                    label=f"Text to synthesise  (max {MAX_TEXT_CHARS} chars)",
                    placeholder="Enter the text you want to speak…",
                    lines=3, max_lines=8,
                )
                text_info = gr.Markdown(f"0 / {MAX_TEXT_CHARS}")

                # Quick-start: pick a bundled sample speaker
                if _SAMPLE_SPEAKERS:
                    sample_dd = gr.Dropdown(
                        SAMPLE_CHOICES, value=SAMPLE_CHOICES[0],
                        label="Quick-start sample speaker  (overrides upload below)",
                    )

                with gr.Row():
                    with gr.Column():
                        ref_audio = gr.Audio(
                            label="Reference audio  (3–30s, WAV/M4A/MP3/FLAC)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        ref_audio_info = gr.Markdown("No file uploaded.")

                ref_text = gr.Textbox(
                    label="Reference transcript  (type the words spoken in the audio above, or use Auto-transcribe)",
                    placeholder="e.g.  Hi, my name is June and I live in Darlington.",
                    lines=3,
                    value="",
                )
                with gr.Row():
                    ref_text_info = gr.Markdown("0 chars")
                    whisper_dd = gr.Dropdown(
                        choices=WHISPER_MODEL_CHOICES,
                        value=WHISPER_MODEL_DEFAULT,
                        label="Whisper model",
                        scale=2,
                    )
                    transcribe_btn = gr.Button(
                        "Auto-transcribe",
                        size="sm", variant="secondary", scale=1,
                    )
                phoneme_preview = gr.Textbox(
                    label="Phoneme check — what espeak-ng sends to the model (verify syllables match your transcript)",
                    interactive=False,
                    lines=2,
                    value="",
                    placeholder="Phonemes appear here when a model is loaded and transcript is typed…",
                )

                streaming_cb = gr.Checkbox(
                    value=True,
                    label="Stream output  (GGUF models only — audio updates in real-time)",
                )
                gen_btn = gr.Button("Generate Speech", variant="primary", size="lg")

                gr.Markdown("### Output")
                output_audio = gr.Audio(
                    label="Synthesised audio", type="numpy", interactive=False,
                )
                stats_md = gr.Markdown("")
                with gr.Row():
                    validate_btn = gr.Button(
                        "Validate output (round-trip test)",
                        size="sm", variant="secondary",
                    )
                validation_md = gr.Markdown("")

        # ── Events ──────────────────────────────────────────────────────────
        load_preset_btn.click(
            fn=load_preset,
            inputs=preset_dd,
            outputs=[backbone_dd, device_dd, codec_dd, temperature, top_k, streaming_cb, preset_status],
        )
        save_preset_btn.click(
            fn=save_preset,
            inputs=[preset_name, backbone_dd, device_dd, codec_dd, temperature, top_k, streaming_cb],
            outputs=save_status,
        )
        clear_btn.click(
            fn=clear_model_cache,
            inputs=[],
            outputs=clear_status,
        )
        load_btn.click(
            fn=load_model,
            inputs=[backbone_dd, device_dd, codec_dd],
            outputs=model_status,
        )
        input_text.change(fn=on_text_change, inputs=input_text, outputs=text_info)
        ref_text.change(fn=on_ref_text_change, inputs=ref_text, outputs=[ref_text_info, phoneme_preview])
        ref_audio.change(fn=on_ref_audio_change, inputs=ref_audio, outputs=ref_audio_info)
        transcribe_btn.click(fn=transcribe_ref_audio, inputs=[ref_audio, whisper_dd], outputs=ref_text)

        if _SAMPLE_SPEAKERS:
            sample_dd.change(
                fn=on_sample_select,
                inputs=sample_dd,
                outputs=[ref_audio, ref_text],
            )

        gen_btn.click(
            fn=generate,
            inputs=[input_text, ref_audio, ref_text, streaming_cb, temperature, top_k],
            outputs=[output_audio, stats_md],
        )
        validate_btn.click(
            fn=validate_output,
            inputs=[output_audio, ref_text],
            outputs=validation_md,
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NeuTTS local UI")
    p.add_argument("--host",  default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port",  type=int, default=7860, help="Port (default: 7860)")
    p.add_argument("--share", action="store_true",   help="Create a public Gradio share link")
    args = p.parse_args()

    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share,
                      theme=gr.themes.Soft())

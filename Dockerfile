FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        cmake \
        ninja-build \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── uv (fast installer) ───────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── App source ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── Python dependencies ───────────────────────────────────────────────────────
# Full install: core + ONNX runtime + Gradio UI + Whisper transcription.
# cmake build step compiles espeak-ng data helpers.
RUN uv pip install --system -e ".[onnx,ui,speech]"

# ── llama-cpp-python (CPU build — no Metal/CUDA on CapRover) ─────────────────
RUN uv pip install --system llama-cpp-python

# ── Runtime config ────────────────────────────────────────────────────────────
# Models download on first use and persist via a CapRover volume at /root/.cache
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]

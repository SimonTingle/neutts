FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg cmake ninja-build build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml ./
COPY neutts/ ./neutts/
COPY neuttsair/ ./neuttsair/
COPY __init__.py ./
COPY CMakeLists.txt ./
COPY README.md ./

RUN uv pip install --system -e ".[onnx,speech]"
RUN uv pip install --system llama-cpp-python fastapi "uvicorn[standard]" soundfile

# ── Bake models into image so restarts are instant ───────────────────────────
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('neuphonic/neutts-nano-q8-gguf'); \
snapshot_download('neuphonic/neucodec-onnx-decoder'); \
print('Models baked in OK')"

# ── App ───────────────────────────────────────────────────────────────────────
COPY server.py ./server.py

ENV NEUTTS_BACKBONE=neuphonic/neutts-nano-q8-gguf
ENV NEUTTS_DEVICE=cpu
ENV NEUTTS_CODEC=neuphonic/neucodec-onnx-decoder
ENV PORT=7860

EXPOSE 7860

CMD ["python", "server.py"]

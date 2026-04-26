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
# Thin frontend install — no torch, no neutts, no llama-cpp-python.
# All inference is delegated to the HF Spaces backend via NEUTTS_BACKEND_URL.
RUN uv pip install --system -r requirements-frontend.txt

# ── Runtime config ────────────────────────────────────────────────────────────
# Set NEUTTS_BACKEND_URL and NEUTTS_API_KEY in CapRover app environment vars.
# Model cache not needed here — models live on the HF Spaces backend.

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]

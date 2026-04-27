FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── App source ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── Python dependencies ───────────────────────────────────────────────────────
# Thin frontend — no torch, no neutts, no llama-cpp-python.
# All inference is delegated to HF Spaces backend via NEUTTS_BACKEND_URL.
RUN uv pip install --system -r requirements-frontend.txt

# ── Runtime config ────────────────────────────────────────────────────────────
# Set in CapRover app environment variables:
#   NEUTTS_BACKEND_URL=https://simontingle-neutts-backend.hf.space
#   NEUTTS_API_KEY=<shared secret>

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]

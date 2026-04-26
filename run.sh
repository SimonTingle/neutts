#!/usr/bin/env bash
# run.sh — NeuTTS local UI launcher
#
# Sets up a uv-managed virtual environment, compiles llama-cpp-python with the
# right hardware backend for your platform, then starts the Gradio UI.
#
# Installed extras:
#   onnx    — ONNX runtime for fast CPU inference
#   ui      — Gradio web interface
#   speech  — openai-whisper for auto-transcription of reference audio
#             (Whisper 'base' model, ~74 MB, downloads on first use)
#
# Usage:
#   ./run.sh                      # start on http://127.0.0.1:7860
#   ./run.sh --port 8080          # custom port
#   ./run.sh --share              # create a public Gradio link
#   ./run.sh --host 0.0.0.0       # listen on all interfaces (LAN access)
#
# Environment variable overrides:
#   NEUTTS_HOST   bind address   (default: 127.0.0.1)
#   NEUTTS_PORT   port           (default: 7860)
#
# To force a fresh Metal recompile of llama-cpp-python:
#   rm .venv/.llama_metal && ./run.sh

set -euo pipefail
IFS=$'\n\t'

# ─── Colour helpers ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
CYN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GRN}[neutts]${NC} $*"; }
warn()  { echo -e "${YLW}[neutts WARN]${NC} $*"; }
error() { echo -e "${RED}[neutts ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }
step()  { echo -e "${CYN}[neutts >>]${NC} $*"; }

# ─── Platform detection ───────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"
IS_MACOS=false
IS_MACOS_ARM=false
[[ "$OS" == "Darwin" ]] && IS_MACOS=true
[[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]] && IS_MACOS_ARM=true

# ─── Argument parsing ─────────────────────────────────────────────────────────

HOST="${NEUTTS_HOST:-127.0.0.1}"
PORT="${NEUTTS_PORT:-7860}"
APP_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --share)           APP_ARGS+=("--share") ;;
        --host=*)          HOST="${arg#*=}"; APP_ARGS+=("--host" "${arg#*=}") ;;
        --host)            : ;;   # handled below with next arg
        --port=*)          PORT="${arg#*=}"; APP_ARGS+=("--port" "${arg#*=}") ;;
        --port)            : ;;
        *)                 APP_ARGS+=("$arg") ;;
    esac
done

# Simple two-argument handling for --host VALUE and --port VALUE
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--host" ]]; then HOST="$arg"; fi
    if [[ "$prev" == "--port" ]]; then PORT="$arg"; fi
    prev="$arg"
done

# ─── Preflight: repo root ─────────────────────────────────────────────────────

preflight() {
    step "Checking repo root..."
    if [[ ! -f "app.py" || ! -f "pyproject.toml" ]]; then
        die "Run this script from the neutts repo root directory."
    fi
    info "Repo root: $(pwd)"
}

# ─── Require: uv ─────────────────────────────────────────────────────────────

require_uv() {
    step "Checking uv..."
    if command -v uv &>/dev/null; then
        info "uv $(uv --version | awk '{print $2}') found."
        return
    fi

    warn "uv not found — installing via the official installer..."

    if ! command -v curl &>/dev/null; then
        die "curl is required to install uv.  Install curl then re-run."
    fi

    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        die "Failed to install uv automatically.
  Install it manually and re-run:
    curl -LsSf https://astral.sh/uv/install.sh | sh
  or visit: https://docs.astral.sh/uv/getting-started/installation/"
    fi

    # Bring uv into PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        die "uv was installed but is not in PATH.
  Open a new shell, then run this script again."
    fi
    info "uv $(uv --version | awk '{print $2}') installed."
}

# ─── Require: Python 3.10–3.13 ───────────────────────────────────────────────

require_python() {
    step "Checking Python (3.10–3.13)..."
    if uv python find ">=3.10,<3.14" &>/dev/null 2>&1; then
        local pyver
        pyver=$(uv python find ">=3.10,<3.14" 2>/dev/null | head -1 || echo "found")
        info "Python found: $pyver"
        return
    fi
    info "Python 3.10-3.13 not found — installing 3.11 via uv..."
    uv python install 3.11 \
        || die "Could not install Python 3.11.
  Install it manually: https://python.org/downloads/"
    info "Python 3.11 installed."
}

# ─── Require: Xcode Command Line Tools (macOS only) ──────────────────────────

require_xcode_tools() {
    $IS_MACOS || return 0
    step "Checking Xcode Command Line Tools..."
    if xcode-select -p &>/dev/null 2>&1; then
        info "Xcode CLT: $(xcode-select -p)"
        return
    fi
    warn "Xcode Command Line Tools not installed — required to compile native extensions."
    info "Launching installer dialog..."
    xcode-select --install 2>/dev/null || true
    die "Re-run this script once the Xcode Command Line Tools installation completes.
  You can monitor progress in System Preferences → Software Update."
}

# ─── Require: cmake (macOS ARM only, for Metal build) ────────────────────────

require_cmake() {
    $IS_MACOS_ARM || return 0
    step "Checking cmake (required for Metal build)..."
    if command -v cmake &>/dev/null; then
        info "cmake $(cmake --version | head -1 | awk '{print $3}') found."
        return
    fi
    warn "cmake not found — required to compile llama-cpp-python with Metal support."
    if command -v brew &>/dev/null; then
        info "Installing cmake via Homebrew..."
        brew install cmake \
            || die "cmake install failed.  Run manually: brew install cmake"
        info "cmake $(cmake --version | head -1 | awk '{print $3}') installed."
    else
        die "cmake is required and Homebrew is not installed.
  Install Homebrew first:  https://brew.sh
  Then run:  brew install cmake"
    fi
}

# ─── Require: espeak-ng ───────────────────────────────────────────────────────

require_espeak() {
    step "Checking espeak-ng..."

    # Apple Silicon Homebrew installs to /opt/homebrew
    if $IS_MACOS; then
        for dir in /opt/homebrew/bin /usr/local/bin; do
            [[ -x "$dir/espeak-ng" ]] && { export PATH="$dir:$PATH"; info "espeak-ng found at $dir/espeak-ng"; return; }
        done
    fi

    if command -v espeak-ng &>/dev/null; then
        info "espeak-ng $(espeak-ng --version 2>&1 | head -1) found."
        return
    fi

    warn "espeak-ng not found."

    if $IS_MACOS; then
        if command -v brew &>/dev/null; then
            info "Installing espeak-ng via Homebrew..."
            brew install espeak-ng \
                || die "Homebrew install failed.  Run manually: brew install espeak-ng"
        else
            die "espeak-ng is required and Homebrew is not installed.
  Install Homebrew first:  https://brew.sh
  Then run:  brew install espeak-ng"
        fi

    elif [[ "$OS" == "Linux" ]]; then
        info "Attempting to install espeak-ng..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y espeak-ng
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y espeak-ng
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm espeak-ng
        else
            die "Cannot auto-install espeak-ng.  Install from:
  https://github.com/espeak-ng/espeak-ng/releases"
        fi

    else
        die "espeak-ng not found.  Install from: https://github.com/espeak-ng/espeak-ng/releases"
    fi
    info "espeak-ng installed."
}

# ─── Require: ffmpeg ─────────────────────────────────────────────────────────
# ffmpeg is required by Gradio 6 to convert M4A/AAC/MP3 uploads for browser
# playback. Without it, non-WAV uploads may return None to Python callbacks.

require_ffmpeg() {
    step "Checking ffmpeg..."

    if ! command -v ffmpeg &>/dev/null; then
        warn "ffmpeg not found — required for M4A/AAC/MP3 audio upload support."
        _install_ffmpeg
        return
    fi

    # Verify it actually runs — a broken dylib (e.g. stale x265 after brew upgrade)
    # will crash with exit code -6 even though the binary exists in PATH.
    if ! ffmpeg -version &>/dev/null 2>&1; then
        warn "ffmpeg is installed but fails to run (likely a broken shared library)."
        _fix_ffmpeg
        return
    fi

    local ver
    ver=$(ffmpeg -version 2>&1 | awk 'NR==1{print $3}')
    info "ffmpeg $ver found and working."
}

_install_ffmpeg() {
    if $IS_MACOS; then
        if command -v brew &>/dev/null; then
            info "Installing ffmpeg via Homebrew..."
            brew install ffmpeg \
                || die "Homebrew install failed.  Run manually: brew install ffmpeg"
            _verify_ffmpeg_runs
        else
            die "ffmpeg is required and Homebrew is not installed.
  Install Homebrew first:  https://brew.sh
  Then run:  brew install ffmpeg"
        fi
    elif [[ "$OS" == "Linux" ]]; then
        info "Attempting to install ffmpeg..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y ffmpeg
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm ffmpeg
        else
            die "Cannot auto-install ffmpeg.  Install from: https://ffmpeg.org/download.html"
        fi
    else
        die "ffmpeg not found.  Install from: https://ffmpeg.org/download.html"
    fi
    info "ffmpeg installed."
}

_fix_ffmpeg() {
    if $IS_MACOS && command -v brew &>/dev/null; then
        warn "Attempting to repair broken Homebrew dependencies..."
        info "Running: brew reinstall x265 && brew reinstall ffmpeg"
        brew reinstall x265  2>/dev/null || warn "x265 reinstall had warnings — continuing."
        brew reinstall ffmpeg || die "ffmpeg reinstall failed.
  Try running: brew doctor
  Then: brew reinstall x265 ffmpeg"
        _verify_ffmpeg_runs
    else
        die "ffmpeg is broken.  Reinstall it and re-run:
  macOS:  brew reinstall x265 ffmpeg
  Linux:  reinstall ffmpeg via your package manager"
    fi
}

_verify_ffmpeg_runs() {
    if ! ffmpeg -version &>/dev/null 2>&1; then
        local stderr_out
        stderr_out=$(ffmpeg -version 2>&1 || true)
        die "ffmpeg is still not working after reinstall.
  Error output:
    $stderr_out
  Try: brew doctor  then re-run this script."
    fi
    local ver
    ver=$(ffmpeg -version 2>&1 | awk 'NR==1{print $3}')
    info "ffmpeg $ver verified working."
}

# ─── Disk space check ─────────────────────────────────────────────────────────

check_disk_space() {
    step "Checking disk space..."
    local required_gb=5
    local avail_kb avail_gb

    if [[ "$OS" == "Darwin" ]]; then
        avail_kb=$(df -k . 2>/dev/null | awk 'NR==2{print $4}') || avail_kb=0
    else
        avail_kb=$(df -k . 2>/dev/null | awk 'NR==2{print $4}') || avail_kb=0
    fi

    avail_gb=$(( avail_kb / 1024 / 1024 ))

    if (( avail_gb < required_gb )); then
        warn "Low disk space: ~${avail_gb} GB available (${required_gb} GB recommended)."
        warn "Model downloads can be 250 MB–2 GB each — free up space if downloads fail."
    else
        info "Disk space: ~${avail_gb} GB available."
    fi
}

# ─── Create virtual environment ───────────────────────────────────────────────

create_venv() {
    step "Setting up virtual environment..."
    if [[ -d ".venv" ]]; then
        info ".venv already exists — skipping creation."
        # Quick sanity check: ensure the venv Python is still executable
        if [[ ! -x ".venv/bin/python" ]]; then
            warn ".venv/bin/python not found or not executable — recreating venv."
            rm -rf .venv
            uv venv --python ">=3.10,<3.14" \
                || die "Failed to recreate virtual environment."
        fi
        return
    fi
    info "Creating virtual environment..."
    uv venv --python ">=3.10,<3.14" \
        || die "Failed to create virtual environment."
    info "Virtual environment created."
}

# ─── Install Python dependencies ──────────────────────────────────────────────

# Stamp file — prevents recompiling llama-cpp-python on every run.
# Delete it to force a fresh Metal recompile: rm .venv/.llama_metal
METAL_STAMP=".venv/.llama_metal"

install_deps() {
    step "Installing Python dependencies (core + ONNX + UI + speech transcription)..."
    uv pip install -e ".[onnx,ui,speech]" \
        || die "Dependency install failed.  Check the error above."
    info "Core dependencies installed (includes openai-whisper for auto-transcription)."
    info "Note: Whisper 'base' model (~74 MB) downloads on first use of 'Auto-transcribe'."

    if $IS_MACOS_ARM; then
        # ── Apple Silicon: compile llama-cpp-python with Metal ───────────────
        if [[ -f "$METAL_STAMP" ]]; then
            info "llama-cpp-python (Metal) already compiled — skipping."
        else
            warn "Compiling llama-cpp-python with Apple Metal support."
            warn "This is a one-time step and typically takes 5–15 minutes."
            echo ""

            CMAKE_ARGS="-DGGML_METAL=ON" \
            uv pip install \
                --no-binary llama-cpp-python \
                --reinstall-package llama-cpp-python \
                llama-cpp-python \
            && touch "$METAL_STAMP" \
            || die "Metal compilation failed.
  Common fixes:
    • Install/update Xcode tools:  xcode-select --install
    • Ensure CMake is installed:   brew install cmake
    • Check Xcode is up to date in the App Store
    • To retry from scratch:       rm .venv/.llama_metal && ./run.sh"
            info "Metal build complete."
        fi

    else
        # ── Other platforms ─────────────────────────────────────────────────
        if command -v nvcc &>/dev/null; then
            info "CUDA detected — compiling llama-cpp-python with CUDA support..."
            CMAKE_ARGS="-DGGML_CUDA=ON" \
            uv pip install --no-binary llama-cpp-python llama-cpp-python \
            || {
                warn "CUDA compilation failed — falling back to CPU-only build."
                uv pip install llama-cpp-python \
                    || die "llama-cpp-python CPU install failed."
            }
        else
            info "Installing llama-cpp-python (CPU build)..."
            uv pip install llama-cpp-python \
                || die "llama-cpp-python install failed.  Check the error above."
        fi
    fi
}

# ─── Port availability check ─────────────────────────────────────────────────

check_port() {
    step "Checking port $PORT..."
    if command -v lsof &>/dev/null; then
        if lsof -iTCP:"$PORT" -sTCP:LISTEN &>/dev/null 2>&1; then
            warn "Port $PORT is already in use.  Gradio will pick the next available port."
        else
            info "Port $PORT is available."
        fi
    else
        info "Port check skipped (lsof not found)."
    fi
}

# ─── Launch UI ────────────────────────────────────────────────────────────────

launch() {
    echo ""
    echo -e "${BLU}────────────────────────────────${NC}"
    info "All checks passed — starting NeuTTS UI."
    info "Open in your browser: http://${HOST}:${PORT}"
    echo ""

    .venv/bin/python app.py --host "$HOST" --port "$PORT" "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BLU}╔══════════════════════════════════╗${NC}"
    echo -e "${BLU}║   NeuTTS — Local Voice Synthesis ║${NC}"
    echo -e "${BLU}╚══════════════════════════════════╝${NC}"
    echo    "  Platform : $OS / $ARCH"
    $IS_MACOS_ARM && echo    "  Hardware : Apple Silicon (Metal GPU available)"
    echo ""

    preflight
    require_uv
    require_python
    require_xcode_tools      # macOS only: CLT needed for any native compilation
    require_cmake            # macOS ARM only: needed for Metal llama.cpp build
    require_espeak
    require_ffmpeg           # installs if missing AND verifies it actually runs
    check_disk_space
    create_venv
    install_deps
    check_port
    launch
}

main "$@"

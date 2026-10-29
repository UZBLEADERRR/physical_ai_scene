#!/bin/bash
# G1 demo on a MacBook: double-click this file in Finder (first time: right-click -> Open).
# Sets up Python (via uv), starts the local Ollama LLM for S5, then opens the demo in the browser.
# Everything runs on this Mac; recordings are kept in data/demo_store/.
set -e
cd "$(dirname "$0")"
MODEL="${S5_MODEL:-qwen2.5:3b}"
PORT="${PORT:-8501}"
say() { printf "\n\033[1;34m[G1]\033[0m %s\n" "$*"; }

# 1. Python 3.11 environment (uv downloads Python itself; no Homebrew Python needed)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "Installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ ! -x .venv/bin/python ]; then
  say "Creating the Python 3.11 environment (first run only, a few minutes)..."
  "$UV" venv -p 3.11 .venv
fi
if [ ! -f .venv/.deps_ok ] || [ requirements-demo.txt -nt .venv/.deps_ok ]; then
  say "Installing packages..."
  VIRTUAL_ENV="$PWD/.venv" "$UV" pip install -r requirements-demo.txt
  touch .venv/.deps_ok
fi

# Open3D's macOS wheel links Homebrew's libusb (and libomp); without them `import open3d` fails.
if [ "$(uname)" = "Darwin" ]; then
  BREW="$(command -v brew || ls /opt/homebrew/bin/brew /usr/local/bin/brew 2>/dev/null | head -1)"
  if ! .venv/bin/python -c "import open3d" >/dev/null 2>&1; then
    if [ -n "$BREW" ]; then
      say "Installing libusb + libomp for Open3D (Homebrew)..."
      "$BREW" install libusb libomp
    else
      say "Open3D needs Homebrew's libusb. Install Homebrew (https://brew.sh), then run: brew install libusb libomp"
      exit 1
    fi
    .venv/bin/python -c "import open3d" || { say "Open3D still does not load; send this window's text to the team."; exit 1; }
  fi
fi

# 2. Local LLM for S5 (Ollama). Without it the demo still runs with the offline rule parser.
if command -v ollama >/dev/null 2>&1 || [ -d /Applications/Ollama.app ]; then
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    say "Starting Ollama..."
    if [ -d /Applications/Ollama.app ]; then open -a Ollama; else (ollama serve >/tmp/g1_ollama.log 2>&1 &); fi
    for _ in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
  fi
  if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"$MODEL\""; then
    say "Downloading the S5 model $MODEL (about 2 GB, first run only)..."
    if command -v ollama >/dev/null 2>&1; then ollama pull "$MODEL"
    else curl -sf http://127.0.0.1:11434/api/pull -d "{\"model\": \"$MODEL\", \"stream\": false}" >/dev/null; fi
  fi
  say "Ollama is running with $MODEL: S5 parser = ollama"
else
  say "Ollama is not installed: S5 uses the offline rule parser."
  say "For the local LLM install it from https://ollama.com/download and run this file again."
fi

# 3. The demo (an older demo still holding the port is stopped first)
for pid in $(lsof -ti tcp:"$PORT" 2>/dev/null); do
  if ps -p "$pid" -o command= | grep -q "streamlit run demo/app.py"; then
    say "Stopping the previous demo (pid $pid) on port $PORT..."
    kill "$pid" && sleep 2
  else
    say "Port $PORT is used by another program; set PORT=8502 and run again."
    exit 1
  fi
done
say "Opening http://localhost:$PORT  (close this window or press Ctrl+C to stop)"
(sleep 6 && open "http://localhost:$PORT") &
export S5_MODEL="$MODEL"
exec .venv/bin/streamlit run demo/app.py --server.port "$PORT" --server.address 127.0.0.1

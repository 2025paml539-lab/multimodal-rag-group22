#!/usr/bin/env bash
# Start of every lab session. Safe to run more than once.
# Usage: bash scripts/lab_start.sh [model ...]   e.g. bash scripts/lab_start.sh qwen3:8b gpt-oss:20b
# Used in every lab session since S2.

OLLAMA="$HOME/ollama012/bin/ollama"
export OLLAMA_MODELS=/tmp/ollama_models        # /tmp is wiped on restart, models are pulled again
mkdir -p "$OLLAMA_MODELS"

if [ ! -x "$OLLAMA" ]; then
  echo "Ollama 0.12.6 not found at $OLLAMA. Follow docs/LAB_SETUP.md step 5." >&2
  exit 1
fi

if ! pgrep -f "ollama012/bin/ollama serve" > /dev/null; then
  nohup "$OLLAMA" serve > "$HOME/ollama012/serve.log" 2>&1 &
  sleep 8
fi

echo "--- Ollama device ---"
grep -i "inference compute" "$HOME/ollama012/serve.log" | tail -1
echo "(expect library=CUDA; library=cpu means the wrong Ollama version)"

for m in "$@"; do
  "$OLLAMA" pull "$m"
done

echo "--- models ---"
"$OLLAMA" list
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "--- disk (keep home under 25 GB) ---"
df -h "$HOME" | tail -1

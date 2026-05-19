#!/usr/bin/env bash
# train.sh — Wrapper for train_wake_word.py
#
# First run (generate TTS + train):
#   ./train.sh "hey haiva" YOUR_AZURE_KEY
#
# Retrain after adding real voice WAVs (no TTS needed):
#   ./train.sh "hey haiva" --skip_tts
#
# Custom threshold / steps:
#   ./train.sh "hey nova" YOUR_AZURE_KEY --max_steps 8000
#
# List mic devices then start listening:
#   python listen.py --list

set -euo pipefail

WAKE_WORD="${1:-hey haiva}"
shift || true   # consume wake_word arg; remaining args passed through

AZURE_KEY=""
EXTRA_ARGS=()

# Pull out --skip_tts or an azure key from remaining args
for arg in "$@"; do
    if [[ "$arg" == --skip_tts ]]; then
        EXTRA_ARGS+=("--skip_tts")
    elif [[ "$arg" == --* ]]; then
        EXTRA_ARGS+=("$arg")
    else
        AZURE_KEY="$arg"
    fi
done

echo "============================================"
echo "  Wake word : \"${WAKE_WORD}\""
echo "============================================"

if [[ -n "$AZURE_KEY" ]]; then
    python3 train_wake_word.py \
        --wake_word "${WAKE_WORD}" \
        --azure_key "${AZURE_KEY}" \
        "${EXTRA_ARGS[@]}"
else
    python3 train_wake_word.py \
        --wake_word "${WAKE_WORD}" \
        "${EXTRA_ARGS[@]}"
fi

SLUG="${WAKE_WORD// /_}"
SLUG="${SLUG,,}"

echo ""
echo "============================================"
echo "  Training complete."
echo "  Model: output_${SLUG}/tflite/${SLUG}_float32.tflite"
echo ""
echo "  Start listening:"
echo "    python listen.py --wake_word \"${WAKE_WORD}\""
echo "============================================"

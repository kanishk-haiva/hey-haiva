#!/bin/bash
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export FLUTTER_HOME=/opt/flutter
export PATH=$FLUTTER_HOME/bin:$PATH
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin

# ── Jenkins parameters (set by Jenkins job) ──────────────────────────────────
AGENT_ID=${agentId}
WORKSPACE_ID=${workspaceId}
ORG_ID=${orgId}
AGENT_NAME=${agentName}
AZURE_KEY=${azureKey}

# Path to the wake word training directory on the Jenkins server
WAKEWORD_DIR=${WAKEWORD_DIR:-/opt/wakeword}
WAKE_WORD=${WAKE_WORD:-"hey haiva"}

echo "=== Build parameters ==="
echo "  AGENT_ID     : $AGENT_ID"
echo "  WORKSPACE_ID : $WORKSPACE_ID"
echo "  ORG_ID       : $ORG_ID"
echo "  AGENT_NAME   : $AGENT_NAME"
echo "  AZURE_KEY    : $AZURE_KEY"
echo "  WAKE_WORD    : $WAKE_WORD"
echo "  WAKEWORD_DIR : $WAKEWORD_DIR"

if [[ -z "$AGENT_ID" || -z "$WORKSPACE_ID" || -z "$ORG_ID" ]]; then
    echo "Error: agentId, workspaceId, or orgId is not set."
    exit 1
fi

java -version
flutter doctor
flutter --version
dart --version

if ! command -v flutter &> /dev/null; then
    echo "Flutter not found. Check FLUTTER_HOME."
    exit 1
fi

# ── Step 1: Train wake word model ─────────────────────────────────────────────
echo ""
echo "=== Step 1: Training wake word model ==="

WAKE_SLUG=$(echo "$WAKE_WORD" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
TRAIN_LOG=/tmp/wake_word_train_${WAKE_SLUG}.log

cd "$WAKEWORD_DIR"

if [[ -z "${AZURE_KEY:-}" ]]; then
    echo "Error: AZURE_KEY is not set. Add it as a Jenkins credential."
    exit 1
fi

# Build Docker image (cached after first run)
docker build -t wakeword-trainer "$WAKEWORD_DIR"

# Run training inside Ubuntu 20.04 container (has OpenSSL 1.1 — required by Azure Speech SDK)
docker run --rm \
    -v "$WAKEWORD_DIR:/app" \
    -e AZURE_KEY="$AZURE_KEY" \
    wakeword-trainer \
    --wake_word "$WAKE_WORD" \
    --azure_key "$AZURE_KEY" \
    --force_retrain \
    2>&1 | tee "$TRAIN_LOG"

# Parse the recommended threshold from "Best threshold: X.XXX  F1=..."
WAKE_THRESHOLD=$(grep "Best threshold:" "$TRAIN_LOG" \
    | tail -1 \
    | sed 's/.*Best threshold: \([0-9.]*\).*/\1/')

if [[ -z "$WAKE_THRESHOLD" ]]; then
    echo "Error: could not parse threshold from training output."
    echo "Last 20 lines of training log:"
    tail -20 "$TRAIN_LOG"
    exit 1
fi
echo "Wake word threshold: $WAKE_THRESHOLD"

# ── Step 2: Copy trained TFLite model into Flutter assets ─────────────────────
echo ""
echo "=== Step 2: Copying TFLite model ==="

TFLITE_SRC="$WAKEWORD_DIR/output_${WAKE_SLUG}/tflite/${WAKE_SLUG}_float32.tflite"
TFLITE_DST="$WORKSPACE/assets/models/${WAKE_SLUG}_float32.tflite"

if [[ ! -f "$TFLITE_SRC" ]]; then
    echo "Error: trained model not found at $TFLITE_SRC"
    exit 1
fi

mkdir -p "$WORKSPACE/assets/models/"
cp "$TFLITE_SRC" "$TFLITE_DST"
echo "Model copied: $TFLITE_SRC → $TFLITE_DST"

# ── Step 3: Build Flutter APK ─────────────────────────────────────────────────
echo ""
echo "=== Step 3: Building Flutter APK ==="

cd "$WORKSPACE"

flutter clean
flutter pub get || { echo "flutter pub get failed"; exit 1; }

dart run rename_app:main all="$AGENT_NAME"

flutter build apk \
    --release \
    --dart-define=AGENT_ID="$AGENT_ID" \
    --dart-define=WAKE_WORD="$WAKE_SLUG" \
    --dart-define=WAKE_THRESHOLD="$WAKE_THRESHOLD" \
    --target=lib/main.dart

# Rename and archive APK
mkdir -p "$WORKSPACE/artifacts"
APK_SRC=build/app/outputs/flutter-apk/app-release.apk
APK_DST="$WORKSPACE/artifacts/app-release.apk"

if [[ ! -f "$APK_SRC" ]]; then
    echo "APK not found at $APK_SRC"
    exit 1
fi

mv "$APK_SRC" "$APK_DST"
echo "APK archived to $APK_DST"

# ── Step 4: Upload APK ────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Uploading APK ==="

DOMAIN_NAME='services.haiva.ai'
UPLOAD_URL="https://$DOMAIN_NAME/v1/filehandling/upload?folder=haiva-release-apks/${ORG_ID}/${WORKSPACE_ID}/${AGENT_ID}"

curl -X POST "$UPLOAD_URL" \
     -F "files=@$APK_DST" \
     -H "pkey: 3fd9d45242a947a43fdb0199bb383c40" \
     --insecure \
     --fail \
     || { echo "APK upload failed"; exit 1; }

echo "APK uploaded to: $UPLOAD_URL"
echo ""
echo "=== Build complete ==="
echo "  Threshold : $WAKE_THRESHOLD"
echo "  Model     : $TFLITE_DST"
echo "  APK       : $APK_DST"

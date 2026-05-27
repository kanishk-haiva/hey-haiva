#!/bin/bash
set -euo pipefail
export WAKEWORD_DIR=`pwd`
export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto.x86_64
export PATH=$JAVA_HOME/bin:$PATH
FLUTTER_APP_DIR=/var/tmp/flutter_app
sudo mkdir -p -m775 $FLUTTER_APP_DIR
sudo chown -R $(whoami):$(whoami) "$FLUTTER_APP_DIR"
export FLUTTER_DIR=$FLUTTER_APP_DIR/haiva-bot

if [ -d "$FLUTTER_DIR/.git" ]; then
    echo "Flutter directory exists. Pulling latest changes..."
    cd "$FLUTTER_DIR"
    git fetch origin
    git checkout hey-haiva-kanishk
    git stash
    git pull origin hey-haiva-kanishk
else
    echo "Cloning repository..."
    ls -ld $FLUTTER_APP_DIR
    git clone -b hey-haiva-kanishk https://${githubPat}:x-oauth-basic@github.com/HaivaInc/haiva-bot.git $FLUTTER_DIR
fi

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
AGENT_VERSION=${agentVersion}
ENVIRONMENT=${ENVIRONMENT}
AGENT_PLATFORM=${AGENT_PLATFORM}

# Path to the wake word training directory on the Jenkins server
WAKEWORD_DIR=${WAKEWORD_DIR:-/opt/wakeword}
WAKE_WORD=${wakeWord}
FLUTTER_DIR=${FLUTTER_DIR:-$WORKSPACE}

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

sudo docker login --username $docker_user --password $dockercreds

# ── Auto-retrain quality thresholds (override in Jenkins env if needed) ───────
# Retrain (without new TTS) whenever the model fails any of these checks:
#
#   MIN_F1             — harmonic mean of precision & recall on the validation set
#   MIN_GAP            — (min_positive) − (max_negative); thin gap = risky in prod
#   MIN_POSITIVE_SCORE — weakest positive must score above this; ensures the model
#                        is confident enough that a safe threshold (>= 0.50) catches
#                        all real wake words
#   MAX_NEGATIVE_SCORE — strongest negative must score below this; ensures no
#                        confusable phrase sneaks past a 0.50 threshold in production
#
# NOTE: we do NOT check the sweep's "best threshold" directly — that value is
# always the LOWEST threshold achieving max F1, so a perfect model with
# min_pos=0.94 and max_neg=0.24 would report threshold=0.25 (correct but
# misleadingly low). Instead we verify the raw score boundaries directly.
MIN_F1=${MIN_F1:-0.85}
MIN_GAP=${MIN_GAP:--0.05}
MIN_POSITIVE_SCORE=${MIN_POSITIVE_SCORE:-0.70}  # all positives must score > 0.70
MAX_NEGATIVE_SCORE=${MAX_NEGATIVE_SCORE:-0.50}  # all negatives must score < 0.50
MAX_RETRAIN=${MAX_RETRAIN:-3}
# Steps per attempt: [initial, retry-1, retry-2, retry-3]
STEPS_SCHEDULE=(5000 7000 9000 12000)

# ── Helpers ───────────────────────────────────────────────────────────────────
parse_metrics() {
    local logfile="$1"
    WAKE_F1=$(grep "Best threshold:" "$logfile" | tail -1 \
        | sed 's/.*F1=\([0-9.]*\).*/\1/')
    WAKE_GAP=$(grep "gap.*=" "$logfile" | tail -1 \
        | sed 's/.*= *\([+-][0-9.]*\).*/\1/')
    WAKE_MIN_POS=$(grep "min_positive" "$logfile" | tail -1 \
        | sed 's/.*= *\([0-9.]*\).*/\1/')
    WAKE_MAX_NEG=$(grep "max_negative" "$logfile" | tail -1 \
        | sed 's/.*= *\([0-9.]*\).*/\1/')
    # Compute production threshold as midpoint between the two score boundaries.
    # e.g. min_pos=0.945 max_neg=0.239 → threshold=0.592
    # This gives clean margin on both sides and is safe to use on device.
    if [[ -n "$WAKE_MIN_POS" && -n "$WAKE_MAX_NEG" ]]; then
        WAKE_THRESHOLD=$(awk -v p="$WAKE_MIN_POS" -v n="$WAKE_MAX_NEG" \
            'BEGIN { printf "%.3f", (p + n) / 2.0 }')
    fi
}

# Returns 0 (success) if F1, gap, min_positive, and max_negative all meet targets.
# Uses awk for float comparison — no python3 dependency on the Jenkins host.
quality_ok() {
    local f1="${WAKE_F1:-0}" gap="${WAKE_GAP:-0}"
    local min_pos="${WAKE_MIN_POS:-0}" max_neg="${WAKE_MAX_NEG:-1}"
    [[ -z "$f1" || "$f1" == "0" ]] && return 1
    awk -v f1="$f1" -v gap="$gap" \
        -v min_pos="$min_pos" -v max_neg="$max_neg" \
        -v min_f1="$MIN_F1" -v min_gap="$MIN_GAP" \
        -v min_pos_thr="$MIN_POSITIVE_SCORE" -v max_neg_thr="$MAX_NEGATIVE_SCORE" \
        'BEGIN { exit (f1 >= min_f1 && gap >= min_gap && min_pos >= min_pos_thr && max_neg <= max_neg_thr) ? 0 : 1 }'
}

# ── Training loop ─────────────────────────────────────────────────────────────
ATTEMPT=0
SKIP_TTS_FLAG=""        # empty on first run; "--skip_tts" from attempt 2 onwards
WAKE_THRESHOLD=""
WAKE_F1=""
WAKE_GAP=""
WAKE_MIN_POS=""
WAKE_MAX_NEG=""

while true; do
    MAX_STEPS=${STEPS_SCHEDULE[$ATTEMPT]:-12000}
    echo ""
    echo "--- Training attempt $((ATTEMPT + 1)) / $((MAX_RETRAIN + 1))"
    echo "    max_steps=$MAX_STEPS  skip_tts=${SKIP_TTS_FLAG:-no}"

    # shellcheck disable=SC2086
    sudo docker run --rm \
        -v "$WAKEWORD_DIR:/app" \
        -e AZURE_KEY="$AZURE_KEY" \
        haivakanishk/wakeword-trainer \
        --wake_word "$WAKE_WORD" \
        --azure_key "$AZURE_KEY" \
        --force_retrain \
        --max_steps "$MAX_STEPS" \
        $SKIP_TTS_FLAG \
        2>&1 | tee "$TRAIN_LOG"

    parse_metrics "$TRAIN_LOG"

    if [[ -z "$WAKE_MIN_POS" || -z "$WAKE_MAX_NEG" ]]; then
        echo "Error: could not parse min_positive / max_negative from training output."
        echo "Last 20 lines of training log:"
        tail -20 "$TRAIN_LOG"
        exit 1
    fi

    echo ""
    echo "  Attempt $((ATTEMPT + 1)) results:"
    echo "    threshold (midpoint) = ${WAKE_THRESHOLD:-?}"
    echo "    min_positive         = ${WAKE_MIN_POS:-?}  (need >= $MIN_POSITIVE_SCORE)"
    echo "    max_negative         = ${WAKE_MAX_NEG:-?}  (need <= $MAX_NEGATIVE_SCORE)"
    echo "    gap                  = ${WAKE_GAP:-?}  (need >= $MIN_GAP)"
    echo "    F1                   = ${WAKE_F1:-?}  (need >= $MIN_F1)"

    if quality_ok; then
        echo "  ✓ Quality OK — proceeding to Flutter build."
        break
    fi

    # Print which check(s) failed
    awk -v f1="${WAKE_F1:-0}" -v gap="${WAKE_GAP:-0}" \
        -v min_pos="${WAKE_MIN_POS:-0}" -v max_neg="${WAKE_MAX_NEG:-1}" \
        -v min_f1="$MIN_F1" -v min_gap="$MIN_GAP" \
        -v min_pos_thr="$MIN_POSITIVE_SCORE" -v max_neg_thr="$MAX_NEGATIVE_SCORE" \
        'BEGIN {
            if (f1      < min_f1)      print "  ✗ F1 too low:               " f1      " < " min_f1
            if (gap     < min_gap)     print "  ✗ Gap too narrow:           " gap     " < " min_gap
            if (min_pos < min_pos_thr) print "  ✗ Weakest positive too low: " min_pos " < " min_pos_thr " (model not confident on positives)"
            if (max_neg > max_neg_thr) print "  ✗ Strongest negative too high: " max_neg " > " max_neg_thr " (a negative sounds too much like the wake word)"
        }'

    if [[ $ATTEMPT -ge $MAX_RETRAIN ]]; then
        echo ""
        echo "  ⚠ WARNING: quality still insufficient after $((MAX_RETRAIN + 1)) attempts."
        echo "  Continuing with best model found (threshold=${WAKE_THRESHOLD:-?})."
        echo "  min_positive=${WAKE_MIN_POS:-?}  max_negative=${WAKE_MAX_NEG:-?}"
        echo "  Consider adding real voice recordings to output_${WAKE_SLUG}/negatives/real/"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    SKIP_TTS_FLAG="--skip_tts"   # reuse existing TTS data — no Azure API cost
    echo "  Retraining without TTS (${STEPS_SCHEDULE[$ATTEMPT]:-12000} steps)…"
done

echo ""
echo "Wake word threshold: $WAKE_THRESHOLD  (F1=${WAKE_F1:-?}  gap=${WAKE_GAP:-?}  min_pos=${WAKE_MIN_POS:-?}  max_neg=${WAKE_MAX_NEG:-?})"

# ── Step 2: Copy trained TFLite model into Flutter assets ─────────────────────
echo ""
echo "=== Step 2: Copying TFLite model ==="

TFLITE_SRC="$WAKEWORD_DIR/output_${WAKE_SLUG}/tflite/${WAKE_SLUG}_float32.tflite"
TFLITE_DST="$FLUTTER_DIR/assets/models/${WAKE_SLUG}_float32.tflite"

if [[ ! -f "$TFLITE_SRC" ]]; then
    echo "Error: trained model not found at $TFLITE_SRC"
    exit 1
fi

mkdir -p "$FLUTTER_DIR/assets/models/"
cp "$TFLITE_SRC" "$TFLITE_DST"
echo "Model copied: $TFLITE_SRC → $TFLITE_DST"

# Register the model in pubspec.yaml so Flutter bundles it in the APK
PUBSPEC="$FLUTTER_DIR/pubspec.yaml"
ASSET_ENTRY="    - assets/models/${WAKE_SLUG}_float32.tflite"
if ! grep -qF "$ASSET_ENTRY" "$PUBSPEC"; then
    # Find line number of LAST existing _float32.tflite entry and insert after it only once
    LAST_LINE=$(grep -n "_float32\.tflite" "$PUBSPEC" | tail -1 | cut -d: -f1)
    if [[ -n "$LAST_LINE" ]]; then
        awk -v n="$LAST_LINE" -v entry="$ASSET_ENTRY" \
            'NR==n{print; print entry; next}1' "$PUBSPEC" > /tmp/pubspec_new.yaml \
            && mv /tmp/pubspec_new.yaml "$PUBSPEC"
        echo "pubspec.yaml updated: added $ASSET_ENTRY"
    else
        echo "WARNING: no tflite model entries found in pubspec.yaml — cannot add $ASSET_ENTRY"
    fi
else
    echo "pubspec.yaml already contains $ASSET_ENTRY — skipping"
fi

# ── Step 3: Build Flutter APK ─────────────────────────────────────────────────
echo ""
echo "=== Step 3: Building Flutter APK ==="
export GRADLE_USER_HOME=/var/tmp/gradle-cache
sudo mkdir -p "$GRADLE_USER_HOME"
sudo chown -R jenkins:jenkins "$GRADLE_USER_HOME"
chmod -R 755 "$GRADLE_USER_HOME"
# Cleanup corrupted transforms cache if exists
rm -rf "$GRADLE_USER_HOME/caches"
rm -rf /home/jenkins/.gradle/caches/8.9/transforms || true

cd "$FLUTTER_DIR"

cat > pubspec_overrides.yaml <<EOF
dependency_overrides:
  azure_speech_recognition_null_safety:
    path: local_packages/azure_speech_recognition_null_safety
  flutter_plugin_android_lifecycle: ^2.0.34
  flutter_sound: ">=9.30.0 <10.0.0"
  flutter_timezone: ^4.1.0
  url_launcher_android: ">=6.3.3 <6.3.10"

EOF

flutter clean
rm -f pubspec.lock
rm -rf ~/.pub-cache
flutter pub get || { echo "flutter pub get failed"; exit 1; }

export GRADLE_USER_HOME=/var/tmp/gradle-cache
sudo mkdir -p "$GRADLE_USER_HOME"
sudo chown -R jenkins:jenkins "$GRADLE_USER_HOME"
chmod -R 755 "$GRADLE_USER_HOME"
# Cleanup corrupted transforms cache if exists
rm -rf "$GRADLE_USER_HOME/caches"
rm -rf /home/jenkins/.gradle/caches/8.9/transforms || true

dart run rename_app:main all="$AGENT_NAME"

flutter build apk \
    --release \
    --dart-define=AGENT_ID="$AGENT_ID" \
    --dart-define=ORG_ID="$ORG_ID" \
    --dart-define=WORKSPACE_ID="$WORKSPACE_ID" \
    --dart-define=AGENT_VERSION="$AGENT_VERSION" \
    --dart-define=ENVIRONMENT="$ENVIRONMENT" \
    --dart-define=AGENT_PLATFORM="$AGENT_PLATFORM" \
    --dart-define=WAKE_WORD="$WAKE_SLUG" \
    --dart-define=WAKE_THRESHOLD="$WAKE_THRESHOLD" \
    --target=lib/main.dart

# Rename and archive APK
mkdir -p "$FLUTTER_DIR/artifacts"
APK_SRC=build/app/outputs/flutter-apk/app-release.apk
APK_DST="$FLUTTER_DIR/artifacts/app-release.apk"

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
UPLOAD_URL="https://$DOMAIN_NAME/v1/filehandling/upload?folder=haiva-release-apks/${ORG_ID}/${WORKSPACE_ID}/${AGENT_ID}/${AGENT_VERSION}/${WAKE_SLUG}"

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

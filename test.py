"""
Test a wake word TFLite model against real-life WAV files.
Uses the same feature extraction pipeline as training (OWW AudioFeatures).

Usage:
    python test.py                          # defaults to hey_nova
    python test.py --wake_word "hey haiva"
    python test.py --model path/to/model.tflite --threshold 0.4
"""

import sys
import glob
import argparse
import numpy as np
import scipy.io.wavfile
import librosa
import tensorflow as tf
from pathlib import Path
from openwakeword.utils import AudioFeatures

CLIP_SAMPLES = 32000    # 2 s @ 16 kHz — must match training
TARGET_RMS   = 3000.0   # int16 scale normalisation

def _wake_word_slug(wake_word: str) -> str:
    return wake_word.strip().lower().replace(" ", "_")

def _resolve_model(wake_word: str, explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            print(f"ERROR: model not found: {p}")
            sys.exit(1)
        return p
    slug = _wake_word_slug(wake_word)
    candidates = [
        Path(f"output_{slug}/tflite/{slug}_float32.tflite"),
        Path(f"output/tflite/{slug}_float32.tflite"),
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"ERROR: No model found for '{wake_word}'. Tried:")
    for c in candidates:
        print(f"  {c}")
    print("Train first or pass --model <path>")
    sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--wake_word", default="hey haiva")
parser.add_argument("--model",     default="", help="Explicit TFLite path")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--dir",       default=".", help="Directory to search for test_*.wav")
args = parser.parse_args()

THRESHOLD   = args.threshold
model_path  = _resolve_model(args.wake_word, args.model)

# ── Load model ────────────────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=str(model_path))
interp.allocate_tensors()
inp_detail  = interp.get_input_details()[0]
out_detail  = interp.get_output_details()[0]
F           = AudioFeatures(device="cpu")

print(f"Wake word: {args.wake_word}")
print(f"Model    : {model_path}")
print(f"Input    : {list(inp_detail['shape'])}  dtype={inp_detail['dtype'].__name__}")
print(f"Output   : {list(out_detail['shape'])}  dtype={out_detail['dtype'].__name__}")
print(f"Threshold: {THRESHOLD}")
print()

# ── Feature extraction (identical to training) ─────────────────────────────
SAMPLE_RATE = 16000

def score_wav(path: str) -> float:
    sr, audio = scipy.io.wavfile.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != SAMPLE_RATE:
        audio_f32 = audio.astype(np.float32) / 32768.0
        audio_f32 = librosa.resample(audio_f32, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16)

    f = audio.astype(np.float64)
    rms = np.sqrt(np.mean(f ** 2))
    if rms > 0:
        audio = np.clip(f * (TARGET_RMS / rms), -32768, 32767).astype(np.int16)

    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    else:
        audio = audio[:CLIP_SAMPLES]

    features = F.embed_clips(audio[np.newaxis, :])          # (1, 16, 96)

    if list(features.shape) != list(inp_detail["shape"]):
        features = np.transpose(features, (0, 2, 1))         # → (1, 96, 16)

    interp.set_tensor(inp_detail["index"], features.astype(np.float32))
    interp.invoke()
    return float(interp.get_tensor(out_detail["index"])[0][0])

# ── Collect WAV files ─────────────────────────────────────────────────────────
wav_files = sorted(glob.glob(str(Path(args.dir) / "test_*.wav")))
if not wav_files:
    print(f"No test_*.wav files found in '{args.dir}'.")
    sys.exit(0)

# ── Run and report ────────────────────────────────────────────────────────────
print(model_path)
print(f"{'File':<35} {'Score':>6}  {'Result'}")
print("-" * 55)

for wav in wav_files:
    try:
        s = score_wav(wav)
        result = "WAKE WORD ✓" if s >= THRESHOLD else "ignored   ✗"
        marker = ">>>" if s >= THRESHOLD else "   "
        print(f"{marker} {Path(wav).name:<32} {s:>6.4f}  {result}")
    except Exception as e:
        print(f"    {Path(wav).name:<32}  ERROR: {e}")

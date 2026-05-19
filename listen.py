"""
Live microphone wake word detector.
Streams audio from the mic in overlapping 2-second windows and scores
each chunk with your TFLite model using the same OWW feature pipeline.

Usage:
    python listen.py                          # defaults to hey_haiva
    python listen.py --wake_word "hey nova"
    python listen.py --model path/to/model.tflite --threshold 0.5
    python listen.py --device 1               # pick a specific mic device index

Requirements (same as test.py, plus PyAudio):
    pip install pyaudio numpy scipy librosa tensorflow openwakeword
"""

import sys
import time
import argparse
import threading
import queue
import wave
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from openwakeword.utils import AudioFeatures

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio not installed. Run:  pip install pyaudio")
    sys.exit(1)

# ── Constants (must match training) ───────────────────────────────────────────
SAMPLE_RATE  = 16000
CLIP_SAMPLES = 32000        # 2 s window
CHUNK_SIZE   = 1600         # 100 ms per read chunk (16 chunks × 100 ms = 1.6 s overlap)
TARGET_RMS   = 3000.0       # int16 normalisation scale
EMA_ALPHA    = 0.35         # EMA smoothing factor (0=no smoothing, 1=no memory)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _slug(wake_word: str) -> str:
    return wake_word.strip().lower().replace(" ", "_")

def _resolve_model(wake_word: str, explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            print(f"ERROR: model not found: {p}")
            sys.exit(1)
        return p
    slug = _slug(wake_word)
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

def list_devices():
    pa = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    print(f"  {'Index':<6} {'Name'}")
    print("  " + "-" * 45)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  {i:<6} {info['name']}")
    pa.terminate()
    print()

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Live wake word detection from microphone")
parser.add_argument("--wake_word",  default="hey haiva",  help="Wake word phrase")
parser.add_argument("--model",      default="",           help="Explicit TFLite model path")
parser.add_argument("--threshold",  type=float, default=0.09, help="Detection threshold (0–1)")
parser.add_argument("--device",     type=int,   default=None, help="Mic device index (omit = default)")
parser.add_argument("--list",       action="store_true",  help="List audio input devices and exit")
parser.add_argument("--cooldown",   type=float, default=1.5,
                    help="Seconds to suppress repeated detections after a trigger")
parser.add_argument("--min_hits",   type=int,   default=2,
                    help="Consecutive frames above threshold (after rising edge) required to trigger")
parser.add_argument("--ema_alpha",  type=float, default=EMA_ALPHA,
                    help="EMA smoothing factor 0–1 (default 0.35)")
args = parser.parse_args()

if args.list:
    list_devices()
    sys.exit(0)

THRESHOLD  = args.threshold
COOLDOWN   = args.cooldown
MIN_HITS   = args.min_hits
EMA_ALPHA  = args.ema_alpha
model_path = _resolve_model(args.wake_word, args.model)

# ── Output dirs ───────────────────────────────────────────────────────────────
_run_ts  = time.strftime("%Y%m%d_%H%M%S")
_slug_ww = _slug(args.wake_word)
LOG_DIR        = Path(f"output_{_slug_ww}/logs");          LOG_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR      = Path(f"output_{_slug_ww}/chunks");        CHUNK_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR_BELOW= Path(f"output_{_slug_ww}/chunks/below");  CHUNK_DIR_BELOW.mkdir(parents=True, exist_ok=True)
LOG_FILE       = LOG_DIR / f"detections_{_run_ts}.log"

# ── Logging / chunk saving ────────────────────────────────────────────────────
def _save_chunk(audio_int16: np.ndarray, score: float, ts: str, below: bool = False) -> Path:
    folder = CHUNK_DIR_BELOW if below else CHUNK_DIR
    prefix = "lo" if below else "hi"
    fname = folder / f"{ts.replace(':', '')}_{prefix}_score{score:.4f}.wav"
    with wave.open(str(fname), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return fname

def _log(line: str):
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")

# ── Load model ────────────────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=str(model_path))
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]
F = AudioFeatures(device="cpu")


print("=" * 55)
print(f"  Wake word : {args.wake_word}")
print(f"  Model     : {model_path}")
print(f"  Threshold : {THRESHOLD}")
print(f"  Cooldown  : {COOLDOWN}s")
print(f"  Min hits  : {MIN_HITS} consecutive frame(s)")
print(f"  Input     : {list(inp_detail['shape'])}  dtype={inp_detail['dtype'].__name__}")
print("=" * 55)
print("\nListening… (Ctrl+C to stop)\n")
_log(f"=== Session started {_run_ts} | wake_word={args.wake_word} | threshold={THRESHOLD} | model={model_path} ===")

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_audio(audio_int16: np.ndarray) -> float:
    """Score a 2-second int16 audio clip and return a confidence in [0, 1]."""
    f = audio_int16.astype(np.float64)
    rms = np.sqrt(np.mean(f ** 2))
    if rms > 0:
        audio_int16 = np.clip(f * (TARGET_RMS / rms), -32768, 32767).astype(np.int16)

    features = F.embed_clips(audio_int16[np.newaxis, :])
    if list(features.shape) != list(inp_detail["shape"]):
        features = np.transpose(features, (0, 2, 1))

    interp.set_tensor(inp_detail["index"], features.astype(np.float32))
    interp.invoke()
    return float(interp.get_tensor(out_detail["index"])[0][0])

# ── Audio capture thread ──────────────────────────────────────────────────────
audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)

def audio_thread(device_index):
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE,
    )
    try:
        while True:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16)
            if not audio_q.full():
                audio_q.put(chunk)
    except Exception as e:
        print(f"\nAudio thread error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

t = threading.Thread(target=audio_thread, args=(args.device,), daemon=True)
t.start()

# ── Sliding window detection loop ─────────────────────────────────────────────
window: deque = deque(maxlen=CLIP_SAMPLES)   # keeps last 2 s of samples
last_trigger = 0.0
ema_score    = 0.0    # exponential moving average of raw scores
was_above    = False  # tracks whether previous frame was above threshold
frames_above = 0      # consecutive frames above threshold since last rising edge

try:
    while True:
        chunk = audio_q.get(timeout=5)
        window.extend(chunk.tolist())

        # Only score when we have a full 2-second window
        if len(window) < CLIP_SAMPLES:
            continue

        audio_clip = np.array(window, dtype=np.int16)
        score = score_audio(audio_clip)

        # EMA smoothing — reduces single-frame noise spikes
        ema_score = EMA_ALPHA * score + (1 - EMA_ALPHA) * ema_score

        now = time.time()
        timestamp = time.strftime("%H:%M:%S")
        in_cooldown = (now - last_trigger) < COOLDOWN

        if ema_score >= THRESHOLD:
            if not was_above:
                # Rising edge — score just crossed threshold from below
                frames_above = 1
                was_above = True
            else:
                frames_above += 1

            if frames_above >= MIN_HITS and not in_cooldown:
                # Trigger: rising edge confirmed + sustained for MIN_HITS frames
                last_trigger = now
                was_above    = False   # require a new rising edge for next trigger
                frames_above = 0
                ema_score    = 0.0    # reset EMA so next event must re-cross
                bar = "█" * int(score * 20)
                wav_path = _save_chunk(audio_clip, score, timestamp)
                msg = (f"[{timestamp}]  WAKE WORD DETECTED!  "
                       f"score={score:.4f}  ema={ema_score:.4f}  "
                       f"|{bar:<20}|  -> {wav_path}")
                print(msg)
                _log(msg)
            else:
                # Building toward trigger or in cooldown — show rising state
                bar = "▒" * int(ema_score * 20)
                state = "cooldown" if in_cooldown else f"rising({frames_above}/{MIN_HITS})"
                msg = (f"[{timestamp}]  {state:<14} "
                       f"score={score:.4f}  ema={ema_score:.4f}  |{bar:<20}|")
                print(msg)
                _log(msg)
        else:
            if was_above:
                # Falling edge — dropped below threshold, reset
                was_above    = False
                frames_above = 0
            bar = "░" * int(score * 20)
            wav_path = _save_chunk(audio_clip, score, timestamp, below=True)
            msg = (f"[{timestamp}]  below          "
                   f"score={score:.4f}  ema={ema_score:.4f}  "
                   f"|{bar:<20}|  -> {wav_path}")
            print(msg)
            _log(msg)

except KeyboardInterrupt:
    _log(f"=== Session stopped {time.strftime('%Y%m%d_%H%M%S')} ===")
    print("\n\nStopped.")
    print(f"\nTo add today's false positives as negatives and retrain:")
    print(f"  mkdir -p output_{_slug_ww}/negatives/real/bg_noise")
    print(f"  cp output_{_slug_ww}/chunks/*.wav output_{_slug_ww}/negatives/real/bg_noise/ 2>/dev/null")
    print(f"  cp output_{_slug_ww}/chunks/below/*.wav output_{_slug_ww}/negatives/real/bg_noise/ 2>/dev/null")
    print(f"  python3 train_wake_word.py --wake_word \"{args.wake_word}\" --skip_tts --force_retrain")
except queue.Empty:
    print("\nNo audio received from microphone — check device index (use --list).")
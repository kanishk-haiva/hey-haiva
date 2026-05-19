"""
Wake Word Training Pipeline — Production Ready
==============================================
Trains a TFLite wake word detector for any phrase.

Quick start
-----------
First run (Azure TTS generates positives and hard negatives):
    python train_wake_word.py --wake_word "hey haiva" --azure_key YOUR_KEY

Retrain after adding real voice samples (no new TTS needed):
    python train_wake_word.py --wake_word "hey haiva" --skip_tts --force_retrain

User-facing directories (only two you ever touch):
    output_{slug}/positives/raw/    ← DROP real "hey haiva" WAV recordings here
    output_{slug}/negatives/real/   ← DROP real non-wake-word WAV recordings here

Everything else is auto-generated — do not edit those folders manually.

Full layout:
    output_{slug}/
        positives/
            raw/          ← YOUR recordings + TTS (auto-written on first run)
            augmented/    ← auto — do not touch
        negatives/
            hard/         ← auto-generated TTS hard negatives — do not touch
            real/         ← YOUR real non-wake-word recordings (optional but recommended)
        model/            ← ONNX intermediate
        tflite/           ← final TFLite model (used by listen.py)
        chunks/           ← listen.py saves detections here
        logs/             ← listen.py logs here
"""

import os
import sys
import time
import types
import shutil
import random
import argparse
import logging

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, AddBackgroundNoise,
)

try:
    from negative_loader import negative_audio_generator
except ImportError:
    negative_audio_generator = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constants (must match listen.py) ──────────────────────────────────────────
SAMPLE_RATE   = 16000
CLIP_SAMPLES  = 32000           # 2 s @ 16 kHz
CLIP_SECONDS  = CLIP_SAMPLES / SAMPLE_RATE   # 2.0

TTS_COUNT      = 80             # TTS samples to generate for the exact wake word
AUGMENT_FACTOR = 12             # augmented copies per raw positive WAV
NEG_VOICES     = 4              # TTS voices per hard-negative phrase

VOICES = [
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural",
    "en-US-BrandonNeural", "en-US-ChristopherNeural",
    "en-GB-SoniaNeural",  "en-GB-RyanNeural",  "en-GB-LibbyNeural",
    "en-AU-NatashaNeural","en-AU-WilliamNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-CA-ClaraNeural",  "en-CA-LiamNeural",
    "en-IE-EmilyNeural",  "en-IE-ConnorNeural",
    "en-NZ-MollyNeural",  "en-SG-LunaNeural",
]
RATE_VALUES  = ["-10%", "-5%", "0%", "+5%", "+10%"]
PITCH_VALUES = ["-5Hz", "0Hz", "+5Hz"]

# Generic phrases that must never trigger for any wake word
_GENERIC_NEGATIVES = [
    "hey google", "hey siri", "alexa", "okay google",
    "good morning", "good night", "how are you", "what time is it",
    "thank you", "excuse me", "I don't know", "see you later",
    "nice to meet you", "did you see that", "I think so",
    "maybe tomorrow", "let me check", "sounds good", "no problem",
    "that's interesting", "I agree", "absolutely",
    "where are we going", "what do you think",
    "play some music", "turn off the lights", "set a timer",
    "what's the weather", "send a message", "take a photo",
    "open the app", "close the door", "call mom",
    "yes", "no", "okay", "hello", "goodbye", "stop", "start", "wait", "go", "help",
]

# Fixed list of alternative leading words (phonetically diverse)
_ALT_PREFIXES = [
    "say", "okay", "hi", "hello", "play", "they", "oh",
    "pray", "day", "way", "pay", "stay",
]

# Fixed list of alternative trailing words
_ALT_SUFFIXES = [
    "nova", "nora", "siri", "alexa", "there", "homer", "rover",
    "lover", "over", "nobody", "cobra", "motor", "yoga", "sofa",
    "toga", "boulder", "colder", "voter", "man", "now", "go",
    "wait", "hi", "you", "noah", "google",
]

# Vowel rotation for phonetic mutations of the trailing word only
_VOWEL_MAP = str.maketrans("aeiou", "eioua")
_CONSONANT_SUBS = {
    "h": ["j", "l", "d", "g"],
    "v": ["b", "f"],
    "n": ["m"],
    "r": ["l"],
    "b": ["v", "p"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def slug(wake_word: str) -> str:
    return wake_word.strip().lower().replace(" ", "_")


def _mutate_trailing(word: str) -> list:
    """Return phonetically-close mutations of a word for use as hard negatives."""
    variants = set()
    variants.add(word.translate(_VOWEL_MAP))
    first = word[0].lower() if word else ""
    for sub in _CONSONANT_SUBS.get(first, []):
        variants.add(sub + word[1:])
    variants.discard(word.lower())
    return [v for v in variants if v]


def build_specific_negatives(wake_word: str) -> list:
    """
    Wake-word-specific hard negatives only — phrases that require TTS because
    they contain words from the wake word itself (prefix swaps, suffix swaps,
    phonetic mutations, partial phrases).

    Generic phrases (hey google, alexa, etc.) are NOT included here — those
    already exist in shared_data/ and are reused without re-generating TTS.
    """
    tokens = wake_word.lower().split()
    phrases = []

    if len(tokens) >= 2:
        rest        = " ".join(tokens[1:])   # e.g. "haiva"
        prefix_part = " ".join(tokens[:-1])  # e.g. "hey"
        last        = tokens[-1]             # e.g. "haiva"

        # 1. Swap leading word: "say haiva", "hi haiva", …
        for alt in _ALT_PREFIXES:
            if alt != tokens[0]:
                phrases.append(f"{alt} {rest}")

        # 2. Swap trailing word: "hey nova", "hey nora", …
        for alt in _ALT_SUFFIXES:
            if alt != last:
                phrases.append(f"{prefix_part} {alt}")

        # 3. Phonetic mutations of the trailing word: "hey heove", "hey jaiva", …
        for mutant in _mutate_trailing(last):
            phrases.append(f"{prefix_part} {mutant}")

        # 4. Partial phrases — the full sequence must be heard, not just one word
        for token in tokens:
            phrases.append(token)           # e.g. "haiva" alone
        phrases.append(prefix_part)         # e.g. "hey" alone

    wake_lower = wake_word.lower()
    seen, result = set(), []
    for p in phrases:
        p = p.strip()
        if p and p not in seen and p != wake_lower:
            seen.add(p)
            result.append(p)

    log.info(f"  Wake-word-specific hard negatives: {len(result)} phrases")
    return result


# ── TTS ───────────────────────────────────────────────────────────────────────
def tts_generate(phrase: str, output_dir: Path,
                 azure_key: str, azure_region: str,
                 count: int = TTS_COUNT) -> list:
    """Generate TTS WAVs for a phrase. Already-existing files are skipped."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        log.error("pip install azure-cognitiveservices-speech")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    generated, new_n, cached_n = [], 0, 0
    for i in tqdm(range(count), desc=f"TTS [{phrase[:30]}]", leave=False):
        out = output_dir / f"tts_{i:05d}.wav"
        if out.exists() and out.stat().st_size > 0:
            generated.append(out)
            cached_n += 1
            continue

        voice = VOICES[new_n % len(VOICES)]
        rate  = random.choice(RATE_VALUES)
        pitch = random.choice(PITCH_VALUES)
        ssml = (
            f"<speak version='1.0' xml:lang='en-US'>"
            f"<voice name='{voice}'>"
            f"<prosody rate='{rate}' pitch='{pitch}'>{phrase}</prosody>"
            f"</voice></speak>"
        )
        audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out))
        synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
        result = synth.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            generated.append(out)
            new_n += 1
        else:
            log.warning(f"TTS failed [{phrase}] sample {i}: {result.reason}")
        time.sleep(0.05)

    log.info(f"  TTS [{phrase}]: {new_n} new + {cached_n} cached = {len(generated)} files")
    return generated


# ── Augmentation ──────────────────────────────────────────────────────────────
def build_augment_pipeline(noise_dir: Optional[Path]) -> Compose:
    transforms = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.6),
        TimeStretch(min_rate=0.80, max_rate=1.20, p=0.6),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Gain(min_gain_db=-8, max_gain_db=8, p=0.5),
    ]
    if noise_dir and noise_dir.exists():
        transforms.append(
            AddBackgroundNoise(sounds_path=str(noise_dir),
                               min_snr_db=5, max_snr_db=30, p=0.5)
        )
        log.info(f"  Background noise augmentation enabled from {noise_dir}")
    return Compose(transforms)


def augment_positives(raw_files: list, output_dir: Path,
                      pipeline: Compose, factor: int) -> list:
    """
    Augment raw positive WAVs into 2-second clips with the wake word placed
    at a RANDOM offset within the window.

    This is the key fix for score variance at inference time: during live
    detection the wake word can appear anywhere in the 2-second sliding
    buffer. Training on clips where the word is always at position 0 creates
    a timing mismatch that causes scores to swing from 0.1 to 0.99 depending
    on when you spoke relative to the window boundary.
    By randomly positioning the word during training, the model learns to
    score the wake word regardless of its position in the 2-second buffer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_files = []
    rng = random.Random(42)

    for src in tqdm(raw_files, desc="Augmenting positives"):
        try:
            audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            log.warning(f"  Skipping {src.name}: {e}")
            continue

        # Normalise loudness
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-6:
            audio = audio * (0.1 / rms)

        # Keep only the utterance (max 1.5 s); silence beyond that is padding
        audio = audio[:int(1.5 * SAMPLE_RATE)]

        for j in range(factor):
            try:
                aug = pipeline(samples=audio.copy(), sample_rate=SAMPLE_RATE)
            except Exception:
                aug = audio.copy()

            # Place word at a random offset so the model sees all timing positions
            max_offset = max(0, CLIP_SAMPLES - len(aug))
            offset = rng.randint(0, max_offset)
            clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            end = min(offset + len(aug), CLIP_SAMPLES)
            clip[offset:end] = aug[:end - offset]

            out = output_dir / f"{src.stem}_aug{j:03d}.wav"
            sf.write(str(out), clip, SAMPLE_RATE)
            out_files.append(out)

    log.info(f"  Augmented: {len(out_files)} clips from {len(raw_files)} raw files")
    return out_files


# ── Feature extraction ────────────────────────────────────────────────────────
def _wav_batch_gen(wav_paths: list, batch_size: int = 32):
    """Yield int16 numpy batches trimmed/zero-padded to CLIP_SAMPLES."""
    import scipy.io.wavfile as wf
    batch = []
    for w in wav_paths:
        try:
            sr, y = wf.read(str(w))
        except Exception as e:
            log.warning(f"  Skipping {Path(w).name}: {e}")
            continue
        if y.ndim > 1:
            y = y[:, 0]
        if y.dtype != np.int16:
            y = np.clip(y.astype(np.float64) * 32768, -32768, 32767).astype(np.int16)
        if len(y) < CLIP_SAMPLES:
            y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
        else:
            y = y[:CLIP_SAMPLES]
        batch.append(y)
        if len(batch) == batch_size:
            yield np.vstack(batch)
            batch = []
    if batch:
        yield np.vstack(batch)


def extract_features(wav_paths: list, npy_path: Path, label: str):
    """Extract OWW audio features, with file-content-based cache invalidation."""
    from openwakeword.utils import compute_features_from_generator

    total_bytes = sum(Path(w).stat().st_size for w in wav_paths)
    cache_key   = f"{len(wav_paths)}:{total_bytes}"
    meta_path   = npy_path.with_suffix(".meta")

    if npy_path.exists():
        cached = meta_path.read_text().strip() if meta_path.exists() else ""
        if cached == cache_key:
            try:
                np.load(str(npy_path), mmap_mode="r")
                log.info(f"  [{label}] Feature cache valid ({len(wav_paths)} WAVs)")
                return
            except Exception:
                pass
        npy_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        log.info(f"  [{label}] Cache stale — re-extracting")

    log.info(f"  [{label}] Extracting features from {len(wav_paths)} WAVs…")
    compute_features_from_generator(
        _wav_batch_gen(wav_paths),
        n_total=len(wav_paths),
        output_file=str(npy_path),
        device="cpu",
        clip_duration=CLIP_SAMPLES,
    )
    meta_path.write_text(cache_key)


def load_examples(npy_path: Path, n_frames: int) -> np.ndarray:
    data = np.load(str(npy_path))
    if data.ndim == 3:
        return data.astype(np.float32)
    examples = [
        data[i:i + n_frames]
        for i in range(0, len(data) - n_frames, n_frames)
    ]
    return np.array(examples, dtype=np.float32)


# ── Batch generators ──────────────────────────────────────────────────────────
def _wrap(arr: np.ndarray, idx: np.ndarray, ptr: int, n: int):
    start = ptr % len(idx)
    sel   = idx[start:start + n]
    if len(sel) < n:
        sel = np.concatenate([sel, idx[:n - len(sel)]])
    return arr[sel], ptr + n


def make_train_gen(pos, hard_neg, easy_neg=None,
                   n_pos=32, n_hard=32, n_easy=16):
    """
    Infinite balanced generator.
    Each batch: n_pos positives + n_hard hard-negatives [+ n_easy easy-negatives].
    Hard negatives (phonetically close phrases) appear in EVERY batch so the
    model always sees the decision boundary.
    """
    import torch
    has_easy = easy_neg is not None and len(easy_neg) > 0

    def _gen():
        pos_idx  = np.arange(len(pos))
        hard_idx = np.arange(len(hard_neg))
        easy_idx = np.arange(len(easy_neg)) if has_easy else None
        dbg = 0

        while True:
            np.random.shuffle(pos_idx)
            np.random.shuffle(hard_idx)
            if has_easy:
                np.random.shuffle(easy_idx)
            pos_ptr = hard_ptr = easy_ptr = 0

            while pos_ptr < len(pos_idx):
                pb = pos[pos_idx[pos_ptr:pos_ptr + n_pos]]
                pos_ptr += n_pos
                if len(pb) == 0:
                    break

                hb, hard_ptr = _wrap(hard_neg, hard_idx, hard_ptr, n_hard)
                parts  = [pb, hb]
                labels = [1.0] * len(pb) + [0.0] * len(hb)

                if has_easy:
                    eb, easy_ptr = _wrap(easy_neg, easy_idx, easy_ptr, n_easy)
                    parts.append(eb)
                    labels += [0.0] * len(eb)

                x    = np.concatenate(parts, axis=0)
                y    = np.array(labels, dtype=np.float32)
                perm = np.random.permutation(len(x))
                dbg += 1
                if dbg <= 3:
                    log.info(f"  [Batch {dbg}] pos={len(pb)} hard-neg={len(hb)}"
                             f" easy-neg={len(eb) if has_easy else 0}")
                yield (
                    torch.tensor(x[perm], dtype=torch.float32),
                    torch.tensor(y[perm], dtype=torch.float32),
                )

    return _gen()


def make_val_batches(pos, neg, n_per=32):
    import torch
    batches = []
    pos_idx, neg_idx = np.arange(len(pos)), np.arange(len(neg))
    pos_ptr = neg_ptr = 0
    while pos_ptr < len(pos_idx):
        pb = pos[pos_idx[pos_ptr:pos_ptr + n_per]]
        pos_ptr += n_per
        if len(pb) == 0:
            break
        nb, neg_ptr = _wrap(neg, neg_idx, neg_ptr, len(pb))
        x = np.concatenate([pb, nb])
        y = np.array([1.0] * len(pb) + [0.0] * len(nb), dtype=np.float32)
        perm = np.random.permutation(len(x))
        batches.append((
            torch.tensor(x[perm], dtype=torch.float32),
            torch.tensor(y[perm], dtype=torch.float32),
        ))
    return batches


# ── OWW model ─────────────────────────────────────────────────────────────────
def _patch_oww_val_guard(oww_model):
    """Guard OWW's early-stopping check against empty val list."""
    try:
        original = oww_model.train_model.__func__
    except AttributeError:
        return oww_model

    def safe_train(self, *a, **kw):
        orig = np.percentile
        def guarded(arr, q, *pa, **pkw):
            if hasattr(arr, "__len__") and len(arr) == 0:
                return 0.0
            return orig(arr, q, *pa, **pkw)
        np.percentile = guarded
        try:
            return original(self, *a, **kw)
        finally:
            np.percentile = orig

    oww_model.train_model = types.MethodType(safe_train, oww_model)
    return oww_model


def train_model(wake_word: str,
                pos_aug_dir: Path,
                neg_hard_dir: Path,
                neg_real_dir: Path,
                shared_neg_dir: Optional[Path],
                shared_bg_aug_dir: Optional[Path],
                features_dir: Path,
                model_out: Path,
                max_steps: int,
                warmup_steps: int,
                hold_steps: int,
                val_split: float = 0.15) -> Path:
    try:
        import torch
        from openwakeword.train import Model
        from openwakeword.utils import AudioFeatures, compute_features_from_generator
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        sys.exit(1)

    # Remove speechbrain to prevent crash chain
    for k in list(sys.modules.keys()):
        if "speechbrain" in k:
            del sys.modules[k]

    features_dir.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    # Collect WAVs
    pos_wavs = list(pos_aug_dir.glob("*.wav"))
    neg_wavs = list(neg_hard_dir.rglob("*.wav"))

    if shared_neg_dir and shared_neg_dir.exists():
        shared_wavs = list(shared_neg_dir.rglob("*.wav"))
        log.info(f"  Shared generic negatives : {len(shared_wavs)} WAVs (reused, no TTS)")
        neg_wavs.extend(shared_wavs)

    if neg_real_dir.exists():
        real_wavs = list(neg_real_dir.rglob("*.wav"))
        log.info(f"  Real negatives           : {len(real_wavs)} WAVs")
        neg_wavs.extend(real_wavs)

    if shared_bg_aug_dir and shared_bg_aug_dir.exists():
        bg_aug_wavs = list(shared_bg_aug_dir.glob("*.wav"))
        log.info(f"  Shared bg noise (aug)    : {len(bg_aug_wavs)} WAVs")
        neg_wavs.extend(bg_aug_wavs)

    log.info(f"  Positive WAVs : {len(pos_wavs)}")
    log.info(f"  Negative WAVs : {len(neg_wavs)} total")

    if not pos_wavs:
        log.error(f"No positive WAVs in {pos_aug_dir}")
        sys.exit(1)
    if not neg_wavs:
        log.error(f"No negative WAVs in {neg_hard_dir}")
        sys.exit(1)

    # Input shape from OWW
    F           = AudioFeatures(device="cpu")
    input_shape = F.get_embedding_shape(CLIP_SECONDS)
    n_frames    = input_shape[0]
    log.info(f"  OWW input shape: {input_shape}")

    # Extract features
    pos_npy = features_dir / "positives.npy"
    neg_npy = features_dir / "negatives.npy"
    extract_features(pos_wavs, pos_npy, "positives")
    extract_features(neg_wavs, neg_npy, "negatives")

    # LibriSpeech easy negatives (if available)
    easy_neg = None
    easy_npy = features_dir / "librispeech.npy"
    libri_dir = Path("./datasets/librispeech")
    noise_dir = Path("./datasets/noise")
    if libri_dir.exists() and negative_audio_generator:
        if not easy_npy.exists():
            log.info("Extracting LibriSpeech easy negatives (one-time, ~5 min)…")
            gen = negative_audio_generator(
                dataset_dirs=[libri_dir],
                noise_dirs=[noise_dir] if noise_dir.exists() else None,
                max_files=10000,
                mix_noise_prob=0.8,
                batch_size=32,
            )
            compute_features_from_generator(
                gen, n_total=10000, output_file=str(easy_npy),
                device="cpu", clip_duration=CLIP_SAMPLES,
            )
        else:
            log.info("  Reusing cached LibriSpeech features")

    all_pos = load_examples(pos_npy, n_frames)
    all_neg = load_examples(neg_npy, n_frames)
    if easy_npy.exists():
        easy_neg = load_examples(easy_npy, n_frames)
        log.info(f"  Easy negatives (LibriSpeech): {len(easy_neg)}")

    np.random.shuffle(all_pos)
    np.random.shuffle(all_neg)

    n_vp = max(1, int(len(all_pos) * val_split))
    n_vn = max(1, int(len(all_neg) * val_split))
    val_pos,  train_pos = all_pos[:n_vp],  all_pos[n_vp:]
    val_neg,  train_neg = all_neg[:n_vn],  all_neg[n_vn:]

    log.info(f"  Train: {len(train_pos)} pos / {len(train_neg)} hard-neg"
             f"  |  Val: {len(val_pos)} pos / {len(val_neg)} neg")

    if len(train_neg) == 0:
        log.error("Zero negative training examples.")
        sys.exit(1)

    train_gen = make_train_gen(train_pos, train_neg, easy_neg,
                               n_pos=32, n_hard=32, n_easy=16)
    val_gen   = make_val_batches(val_pos, val_neg, n_per=32)

    oww = _patch_oww_val_guard(
        Model(n_classes=1, input_shape=input_shape, model_type="dnn",
              layer_dim=256, seconds_per_example=CLIP_SECONDS)
    )

    log.info(f"  Training: {max_steps} steps  warmup={warmup_steps}  hold={hold_steps}")
    try:
        oww.train_model(X=train_gen, X_val=val_gen, max_steps=max_steps,
                        warmup_steps=warmup_steps, hold_steps=hold_steps)
    except TypeError:
        log.warning("  X_val not supported — training without validation set")
        oww.train_model(X=train_gen, max_steps=max_steps,
                        warmup_steps=warmup_steps, hold_steps=hold_steps)

    model_name = slug(wake_word)
    oww.export_model(model=oww.model, model_name=model_name, output_dir=str(model_out))
    onnx_path = model_out / f"{model_name}.onnx"
    log.info(f"  ONNX saved: {onnx_path}")
    return onnx_path


# ── TFLite export ─────────────────────────────────────────────────────────────
def export_tflite(onnx_path: Path, tflite_dir: Path) -> Path:
    import subprocess
    tflite_dir.mkdir(parents=True, exist_ok=True)
    onnx2tf = Path(sys.executable).parent / "onnx2tf"
    if not onnx2tf.exists():
        log.error("onnx2tf not found. Run: pip install onnx2tf")
        sys.exit(1)
    log.info(f"  Converting {onnx_path.name} → TFLite…")
    subprocess.run([str(onnx2tf), "-i", str(onnx_path), "-o", str(tflite_dir)], check=True)
    out = tflite_dir / (onnx_path.stem + "_float32.tflite")
    log.info(f"  TFLite: {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return out


# ── Validation ────────────────────────────────────────────────────────────────
def validate(tflite_path: Path, pos_aug_dir: Path,
             neg_hard_dir: Path, neg_real_dir: Path,
             shared_neg_dir: Optional[Path] = None,
             shared_bg_aug_dir: Optional[Path] = None):
    try:
        import tensorflow as tf
        from openwakeword.utils import AudioFeatures
        import scipy.io.wavfile
    except ImportError:
        log.warning("TF not available — skipping validation")
        return

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    F   = AudioFeatures(device="cpu")

    log.info(f"  Model input : {list(inp['shape'])}  {inp['dtype'].__name__}")
    log.info(f"  Model output: {list(out['shape'])}  {out['dtype'].__name__}")

    def score(wav_path: Path) -> float:
        sr, audio = scipy.io.wavfile.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            f32 = audio.astype(np.float32) / 32768.0
            f32 = librosa.resample(f32, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio = np.clip(f32 * 32768.0, -32768, 32767).astype(np.int16)
        f   = audio.astype(np.float64)
        rms = np.sqrt(np.mean(f ** 2))
        if rms > 0:
            audio = np.clip(f * (3000.0 / rms), -32768, 32767).astype(np.int16)
        if len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        audio = audio[:CLIP_SAMPLES]
        feats = F.embed_clips(audio[np.newaxis, :])
        if list(feats.shape) != list(inp["shape"]):
            feats = np.transpose(feats, (0, 2, 1))
        interp.set_tensor(inp["index"], feats.astype(np.float32))
        interp.invoke()
        return float(interp.get_tensor(out["index"])[0][0])

    def report(label: str, paths, expect_high: bool, limit: int = 80):
        results = []
        for p in list(paths)[:limit]:
            try:
                results.append((Path(p), score(Path(p))))
            except Exception as e:
                log.warning(f"  skip {Path(p).name}: {e}")
        if not results:
            log.warning(f"  {label}: no samples found")
            return []
        scores = [s for _, s in results]
        correct = sum(1 for s in scores if (s > 0.5) == expect_high)
        log.info(f"  {label}: {correct}/{len(scores)} correct  "
                 f"mean={np.mean(scores):.3f}  [{np.min(scores):.3f}–{np.max(scores):.3f}]")
        worst = sorted(results, key=lambda x: x[1] if expect_high else -x[1])[:5]
        for p, s in worst:
            sym = "✓" if (s > 0.5) == expect_high else "✗"
            log.info(f"    {sym} {s:.4f}  {p.name}")
        return scores

    log.info("")
    log.info("=" * 60)
    log.info("VALIDATION REPORT")
    pos_scores = report("POSITIVES (must trigger)",  pos_aug_dir.glob("*.wav"), True)
    neg_wavs = list(neg_hard_dir.rglob("*.wav"))
    if shared_neg_dir and shared_neg_dir.exists():
        shared_wavs = list(shared_neg_dir.rglob("*.wav"))
        log.info(f"  Shared negatives : {len(shared_wavs)} WAVs")
        neg_wavs += shared_wavs
    if neg_real_dir.exists():
        real_wavs = list(neg_real_dir.rglob("*.wav"))
        log.info(f"  Real negatives   : {len(real_wavs)} WAVs")
        neg_wavs += real_wavs
    if shared_bg_aug_dir and shared_bg_aug_dir.exists():
        bg_wavs = list(shared_bg_aug_dir.glob("*.wav"))
        log.info(f"  Shared bg (aug)  : {len(bg_wavs)} WAVs")
        neg_wavs += bg_wavs
    log.info(f"  Hard negatives   : {len(list(neg_hard_dir.rglob('*.wav')))} WAVs")
    log.info(f"  Total to score   : {len(neg_wavs)} WAVs")
    neg_scores = report("NEGATIVES (must not trigger)", iter(neg_wavs), False, limit=len(neg_wavs))

    if not pos_scores or not neg_scores:
        return

    min_pos = float(np.min(pos_scores))
    max_neg = float(np.max(neg_scores))
    gap     = min_pos - max_neg
    mid     = (min_pos + max_neg) / 2.0

    log.info("")
    log.info(f"  min_positive = {min_pos:.3f}")
    log.info(f"  max_negative = {max_neg:.3f}")
    log.info(f"  gap          = {gap:+.3f}")

    if gap > 0.3:
        log.info(f"  ✓ Strong separation")
    elif gap > 0.0:
        log.info(f"  △ Marginal separation — consider more training steps or real-voice data")
    else:
        log.warning(f"  ✗ Scores overlap by {-gap:.3f} — retrain with more data or steps")

    # Sweep thresholds to find best F1
    labeled = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    best = {"f1": -1.0, "thresh": 0.5, "prec": 0.0, "rec": 0.0}
    for t in np.arange(0.05, 0.96, 0.025):
        tp   = sum(1 for s, l in labeled if l == 1 and s >= t)
        fp   = sum(1 for s, l in labeled if l == 0 and s >= t)
        fn   = sum(1 for s, l in labeled if l == 1 and s < t)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best["f1"]:
            best = {"f1": f1, "thresh": round(float(t), 3),
                    "prec": prec, "rec": rec}

    log.info(f"  Best threshold: {best['thresh']}  "
             f"F1={best['f1']:.3f}  Prec={best['prec']:.3f}  Rec={best['rec']:.3f}")
    log.info("")
    log.info(f"  → Run detector:")
    log.info(f"      python listen.py --threshold {best['thresh']}")
    log.info("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train a production wake word detector for any phrase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run — generate TTS and train:
  python train_wake_word.py --wake_word "hey haiva" --azure_key YOUR_KEY

  # Retrain after adding real voice WAVs to positives/raw/:
  python train_wake_word.py --wake_word "hey haiva" --skip_tts --force_retrain

  # Train a different wake word:
  python train_wake_word.py --wake_word "hey nova" --azure_key YOUR_KEY
        """
    )
    parser.add_argument("--wake_word",      default="hey haiva",
                        help="The wake word phrase (default: hey haiva)")
    parser.add_argument("--azure_key",      default="",
                        help="Azure Cognitive Services key (required unless --skip_tts)")
    parser.add_argument("--azure_region",   default="eastus",
                        help="Azure region (default: eastus)")
    parser.add_argument("--skip_tts",       action="store_true",
                        help="Skip TTS; use existing files in positives/raw/ and negatives/hard/")
    parser.add_argument("--force_retrain",  action="store_true",
                        help="Re-augment and retrain even if a model already exists")
    parser.add_argument("--tts_count",      type=int, default=TTS_COUNT,
                        help=f"TTS samples for exact wake word (default: {TTS_COUNT})")
    parser.add_argument("--augment_factor", type=int, default=AUGMENT_FACTOR,
                        help=f"Augmentations per raw positive WAV (default: {AUGMENT_FACTOR})")
    parser.add_argument("--neg_voices",     type=int, default=NEG_VOICES,
                        help=f"TTS voices per hard-negative phrase (default: {NEG_VOICES})")
    parser.add_argument("--max_steps",      type=int, default=5000,
                        help="Training steps (default: 5000)")
    parser.add_argument("--warmup_steps",   type=int, default=500,
                        help="LR warmup steps (default: 500)")
    parser.add_argument("--hold_steps",     type=int, default=1500,
                        help="LR hold steps (default: 1500)")
    parser.add_argument("--shared_dir",     default="./shared_data",
                        help="Shared generic negatives dir reused across wake words "
                             "(default: ./shared_data)")
    parser.add_argument("--noise_dir",      default="",
                        help="Directory of WAVs used as background noise during augmentation")
    parser.add_argument("--output_dir",     default="",
                        help="Root output directory (default: ./output_{slug})")
    args = parser.parse_args()

    ww    = args.wake_word.strip()
    _slug = slug(ww)
    root  = Path(args.output_dir) if args.output_dir else Path(f"./output_{_slug}")

    # Directory layout
    pos_raw_dir    = root / "positives" / "raw"
    pos_aug_dir    = root / "positives" / "augmented"
    neg_hard_dir   = root / "negatives" / "hard"
    neg_real_dir   = root / "negatives" / "real"
    neg_real_aug_dir   = root / "negatives" / "real_augmented"
    shared_bg_aug_dir  = root / "negatives" / "shared_bg_augmented"
    shared_neg_dir = Path(args.shared_dir) / "true_negatives"
    shared_bg_dir  = Path(args.shared_dir) / "bg_noise"
    features_dir   = root / "model" / "features"
    model_dir      = root / "model"
    tflite_dir     = root / "tflite"
    noise_dir      = Path(args.noise_dir) if args.noise_dir else None

    tflite_path  = tflite_dir / f"{_slug}_float32.tflite"
    onnx_path    = model_dir  / f"{_slug}.onnx"

    log.info("=" * 60)
    log.info(f"  Wake word : '{ww}'")
    log.info(f"  Output    : {root.resolve()}")
    log.info(f"  TFLite    : {tflite_path}")
    log.info("=" * 60)

    # ── Step 1: Positive WAVs (TTS or existing) ───────────────────────────────
    pos_raw_dir.mkdir(parents=True, exist_ok=True)
    existing_pos = list(pos_raw_dir.glob("*.wav"))

    if existing_pos:
        log.info(f"Step 1: {len(existing_pos)} positive WAVs in {pos_raw_dir} — skipping TTS")
    elif args.skip_tts:
        log.error(f"No WAVs in {pos_raw_dir} and --skip_tts is set.")
        log.error(f"Add recordings to {pos_raw_dir} or remove --skip_tts.")
        sys.exit(1)
    else:
        if not args.azure_key:
            log.error("Provide --azure_key to generate TTS, or add WAVs to "
                      f"{pos_raw_dir} and re-run with --skip_tts.")
            sys.exit(1)
        log.info(f"Step 1: Generating {args.tts_count} TTS samples for '{ww}'…")
        tts_generate(ww, pos_raw_dir, args.azure_key, args.azure_region, args.tts_count)
        existing_pos = list(pos_raw_dir.glob("*.wav"))

    # ── Step 2: Augment positives ─────────────────────────────────────────────
    expected_aug = len(existing_pos) * args.augment_factor
    existing_aug = list(pos_aug_dir.glob("*.wav")) if pos_aug_dir.exists() else []

    if len(existing_aug) >= expected_aug and not args.force_retrain:
        log.info(f"Step 2: Augmentation up to date ({len(existing_aug)} files)")
        pipeline = None
    else:
        log.info(f"Step 2: Augmenting {len(existing_pos)} positives × {args.augment_factor}…")
        if pos_aug_dir.exists():
            shutil.rmtree(pos_aug_dir)
        pipeline = build_augment_pipeline(noise_dir)
        augment_positives(existing_pos, pos_aug_dir, pipeline, args.augment_factor)

    # ── Step 2b: Augment real negatives ───────────────────────────────────────
    # Real-voice negatives (e.g. "haiva" alone, "hey" alone, bg noise clips) are
    # few in number. Without augmentation the model sees each one exactly once,
    # which is nowhere near enough to generalise. We augment them the same way
    # as positives — random position in the 2s window, varied pitch/speed/noise.
    real_neg_wavs = list(neg_real_dir.rglob("*.wav")) if neg_real_dir.exists() else []
    if real_neg_wavs:
        expected_neg_aug = len(real_neg_wavs) * args.augment_factor
        existing_neg_aug = list(neg_real_aug_dir.glob("*.wav")) if neg_real_aug_dir.exists() else []
        if len(existing_neg_aug) >= expected_neg_aug and not args.force_retrain:
            log.info(f"Step 2b: Real-negative augmentation up to date ({len(existing_neg_aug)} files)")
        else:
            log.info(f"Step 2b: Augmenting {len(real_neg_wavs)} real negatives × {args.augment_factor}…")
            if neg_real_aug_dir.exists():
                shutil.rmtree(neg_real_aug_dir)
            if pipeline is None:
                pipeline = build_augment_pipeline(noise_dir)
            augment_positives(real_neg_wavs, neg_real_aug_dir, pipeline, args.augment_factor)
    else:
        log.info("Step 2b: No real negatives to augment")

    # ── Step 2c: Augment shared bg noise (once per wake word) ─────────────────
    shared_bg_wavs = list(shared_bg_dir.rglob("*.wav")) if shared_bg_dir.exists() else []
    if shared_bg_wavs:
        expected_bg_aug = len(shared_bg_wavs) * args.augment_factor
        existing_bg_aug = list(shared_bg_aug_dir.glob("*.wav")) if shared_bg_aug_dir.exists() else []
        if len(existing_bg_aug) >= expected_bg_aug and not args.force_retrain:
            log.info(f"Step 2c: Shared bg noise augmentation up to date ({len(existing_bg_aug)} files)")
        else:
            log.info(f"Step 2c: Augmenting {len(shared_bg_wavs)} shared bg noise files × {args.augment_factor}…")
            if shared_bg_aug_dir.exists():
                shutil.rmtree(shared_bg_aug_dir)
            if pipeline is None:
                pipeline = build_augment_pipeline(noise_dir)
            augment_positives(shared_bg_wavs, shared_bg_aug_dir, pipeline, args.augment_factor)
    else:
        log.info("Step 2c: No shared bg noise found in shared_data/bg_noise/")

    # ── Step 3: Hard negatives ────────────────────────────────────────────────
    # All hard negatives (generic + wake-word-specific) are stored in shared_data/
    # true_negatives/ so they are generated once and reused for every wake word.
    specific_phrases = build_specific_negatives(ww)

    shared_count = len(list(shared_neg_dir.rglob("*.wav"))) if shared_neg_dir.exists() else 0
    log.info(f"Step 3: Shared negatives: {shared_count} WAVs in {shared_neg_dir}")

    if args.skip_tts:
        if shared_count == 0:
            log.warning("  No negatives found at all. Model will have a high false-positive rate.")
        else:
            log.info(f"  --skip_tts → using {shared_count} WAVs from shared_data/")
    else:
        if not args.azure_key:
            log.warning("Step 3: No --azure_key — skipping hard-negative TTS.")
            log.warning("  Shared negatives will still be used from shared_data/.")
        else:
            shared_neg_dir.mkdir(parents=True, exist_ok=True)
            to_generate = [
                p for p in specific_phrases
                if len(list((shared_neg_dir / p.replace(" ", "_")).glob("*.wav"))) < args.neg_voices
            ]
            skipped = len(specific_phrases) - len(to_generate)
            if skipped:
                log.info(f"  {skipped}/{len(specific_phrases)} phrases already in shared_data — skipping TTS")
            if to_generate:
                log.info(f"  Generating TTS for {len(to_generate)} new phrases × {args.neg_voices} voices…")
                for phrase in tqdm(to_generate, desc="Specific hard negatives"):
                    phrase_dir = shared_neg_dir / phrase.replace(" ", "_")
                    tts_generate(phrase, phrase_dir, args.azure_key, args.azure_region,
                                 args.neg_voices)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    if tflite_path.exists() and not args.force_retrain:
        log.info(f"Step 4: Model already exists — skipping training.")
        log.info(f"  Use --force_retrain to retrain.")
    else:
        log.info("Step 4: Training…")
        onnx_out = train_model(
            wake_word=ww,
            pos_aug_dir=pos_aug_dir,
            neg_hard_dir=neg_hard_dir,
            neg_real_dir=neg_real_aug_dir,
            shared_neg_dir=shared_neg_dir,
            shared_bg_aug_dir=shared_bg_aug_dir if shared_bg_wavs else None,
            features_dir=features_dir,
            model_out=model_dir,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            hold_steps=args.hold_steps,
        )

        # ── Step 5: Export TFLite ─────────────────────────────────────────────
        log.info("Step 5: Exporting TFLite…")
        tflite_path = export_tflite(onnx_out, tflite_dir)

    # ── Step 6: Validate ──────────────────────────────────────────────────────
    log.info("Step 6: Validating…")
    validate(tflite_path, pos_aug_dir, neg_hard_dir, neg_real_aug_dir, shared_neg_dir,
             shared_bg_aug_dir if shared_bg_wavs else None)

    log.info("")
    log.info("=== Pipeline complete ===")
    log.info(f"  Wake word : {ww}")
    log.info(f"  TFLite    : {tflite_path}")
    log.info(f"  Run: python listen.py --wake_word \"{ww}\"")


if __name__ == "__main__":
    main()

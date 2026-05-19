import soundfile as sf
import librosa
from pathlib import Path
import random
import numpy as np

def find_audio_files(root, exts=('.wav', '.flac')):
    return [p for p in Path(root).rglob('*') if p.suffix.lower() in exts]

def negative_audio_generator(
    dataset_dirs,
    noise_dirs=None,
    sample_rate=16000,
    duration=2.0,
    normalize=True,
    mix_noise_prob=0.5,
    max_files=None,
    batch_size=32
):
    files = []
    batch = []
    for d in dataset_dirs:
        files.extend(find_audio_files(d))
    if max_files:
        files = random.sample(files, min(max_files, len(files)))
    if noise_dirs:
        noise_files = []
        for nd in noise_dirs:
            noise_files.extend(find_audio_files(nd))
    else:
        noise_files = []

    for f in files:
        try:
            audio, sr = sf.read(f)
        except Exception:
            continue
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Normalize
        if normalize:
            audio = audio / (abs(audio).max() + 1e-8)
        # Pad/trim
        target_len = int(sample_rate * duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        # Mix noise
        if noise_files and random.random() < mix_noise_prob:
            noise_path = random.choice(noise_files)
            try:
                noise, nsr = sf.read(noise_path)
            except Exception:
                noise = None
            if noise is not None:
                if nsr != sample_rate:
                    noise = librosa.resample(noise, orig_sr=nsr, target_sr=sample_rate)
                if noise.ndim > 1:
                    noise = noise.mean(axis=1)
                if len(noise) < target_len:
                    noise = np.pad(noise, (0, target_len - len(noise)))
                else:
                    noise = noise[:target_len]
                snr_db = random.uniform(0, 10)
                alpha = 10 ** (-snr_db / 20)
                audio = (audio + alpha * noise) / (1 + alpha)
            
        audio_int16 = (audio * 32767).astype(np.int16)
        batch.append(audio_int16)
        if len(batch) == batch_size:
            yield np.vstack(batch)
            batch = []
    
    if batch:
        yield np.vstack(batch)
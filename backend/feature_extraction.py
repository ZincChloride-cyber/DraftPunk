"""
feature_extraction.py
─────────────────────
Extracts audio features using Librosa for genre classification.

Features extracted (per segment):
  • MFCC           — 40 coefficients × (mean + std) = 80 values
  • Spectral Centroid  — mean + std = 2 values
  • Spectral Roll-off  — mean + std = 2 values
  • Spectral Bandwidth — mean + std = 2 values
  • Zero-Crossing Rate — mean + std = 2 values
  • Chroma Features    — 12 bins × (mean + std) = 24 values
  • Tempo              — 1 value
                                              Total = 113 values
"""

from __future__ import annotations

import numpy as np
import librosa

from config import (
    AUDIO_DURATION,
    SAMPLE_RATE,
    N_MFCC,
    N_CHROMA,
    HOP_LENGTH,
    N_FFT,
    SILENCE_THRESHOLD,
    SEGMENT_DURATION,
)


def load_and_preprocess(
    file_path: str,
    *,
    max_duration: int | None = None,
) -> np.ndarray:
    """
    Load an audio file, resample, normalise, and trim silence.

    Parameters
    ----------
    file_path : str
        Path to the audio file (WAV / MP3).

    Returns
    -------
    np.ndarray
        Pre-processed mono audio signal at SAMPLE_RATE Hz.
    """
    # Load and resample to target rate
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Normalise volume to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    # Trim silence from start and end
    signal, _ = librosa.effects.trim(signal, top_db=abs(SILENCE_THRESHOLD))

    # For uploads longer than GTZAN clips, use the middle section (matches training data)
    if max_duration is not None:
        max_samples = max_duration * SAMPLE_RATE
        if len(signal) > max_samples:
            start = (len(signal) - max_samples) // 2
            signal = signal[start : start + max_samples]

    return signal


def split_into_segments(signal: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """
    Split a signal into fixed-length segments.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    list[np.ndarray]
        List of signal segments, each SEGMENT_DURATION seconds long.
    """
    segment_samples = sr * SEGMENT_DURATION
    segments = []

    for start in range(0, len(signal), segment_samples):
        segment = signal[start : start + segment_samples]
        # Only keep full-length segments
        if len(segment) == segment_samples:
            segments.append(segment)

    return segments


def extract_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a feature vector from a single audio segment.

    Parameters
    ----------
    signal : np.ndarray
        Audio segment (mono, at SAMPLE_RATE Hz).
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        1-D feature vector (113 values).
    """
    features = []

    # ── MFCC (40 coefficients → 80 values) ────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # ── Spectral Centroid (2 values) ──────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # ── Spectral Roll-off (2 values) ─────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # ── Spectral Bandwidth (2 values) ────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # ── Zero-Crossing Rate (2 values) ────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=HOP_LENGTH)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # ── Chroma Features (12 × 2 = 24 values) ────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=signal, sr=sr, n_chroma=N_CHROMA, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # ── Tempo (1 value) ──────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr, hop_length=HOP_LENGTH)
    # librosa may return an array; extract scalar
    tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    features.append(tempo_val)

    return np.array(features, dtype=np.float64)


def extract_features_from_file(
    file_path: str,
    *,
    for_prediction: bool = False,
) -> list[np.ndarray]:
    """
    Full pipeline: load → preprocess → segment → extract features.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    for_prediction : bool
        When True, crop long tracks to the center AUDIO_DURATION window (GTZAN-style)
        and pad short clips to SEGMENT_DURATION so features match training.

    Returns
    -------
    list[np.ndarray]
        One feature vector per segment.
    """
    signal = load_and_preprocess(
        file_path,
        max_duration=AUDIO_DURATION if for_prediction else None,
    )
    segments = split_into_segments(signal)

    # Pad short clips to the same length used during training
    if not segments:
        segment_samples = SAMPLE_RATE * SEGMENT_DURATION
        if len(signal) < segment_samples:
            signal = librosa.util.fix_length(signal, size=segment_samples)
        segments = [signal]

    return [extract_features(seg) for seg in segments]


def extract_waveform_for_display(
    signal: np.ndarray,
    *,
    target_samples: int = 600,
) -> list[float]:
    """Downsample amplitude for a line waveform (same signal as mel spectrogram)."""
    block = max(1, len(signal) // target_samples)
    samples: list[float] = []
    for i in range(target_samples):
        start = i * block
        chunk = signal[start : start + block]
        if len(chunk) == 0:
            samples.append(0.0)
            continue
        samples.append(float((np.min(chunk) + np.max(chunk)) / 2))
    peak = max((abs(v) for v in samples), default=0.0)
    if peak > 0:
        samples = [v / peak for v in samples]
    return [round(v, 4) for v in samples]


def extract_mel_spectrogram_for_display(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    *,
    max_time_bins: int = 400,
    max_mel_bins: int = 80,
    fmax_hz: float = 8000,
) -> dict:
    """
    Compute a mel power spectrogram (dB) for UI visualization.

    Returns a downsampled 2-D matrix plus metadata so the frontend can
    label axes without re-running librosa.
    """
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=128,
        fmax=fmax_hz,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_step = max(1, mel_db.shape[0] // max_mel_bins)
    time_step = max(1, mel_db.shape[1] // max_time_bins)
    mel_db = mel_db[::mel_step, ::time_step]

    return {
        "values": np.round(mel_db, 2).tolist(),
        "n_mels": int(mel_db.shape[0]),
        "n_frames": int(mel_db.shape[1]),
        "duration": round(len(signal) / sr, 2),
        "sample_rate": sr,
        "hop_length": HOP_LENGTH,
        "fmax_hz": fmax_hz,
        "db_min": round(float(mel_db.min()), 2),
        "db_max": round(float(mel_db.max()), 2),
    }


def extract_features_for_prediction(file_path: str) -> list[np.ndarray]:
    """
    Extract per-segment feature vectors for prediction.

    The model was trained on individual 10-second segments, not averaged
    features across a whole song — callers should predict per segment and
    aggregate probabilities.

    Returns
    -------
    list[np.ndarray]
        One 113-dimensional feature vector per segment.
    """
    return extract_features_from_file(file_path, for_prediction=True)

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

import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    N_MFCC,
    N_CHROMA,
    HOP_LENGTH,
    N_FFT,
    SILENCE_THRESHOLD,
    SEGMENT_DURATION,
)


def load_and_preprocess(file_path: str) -> np.ndarray:
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


def extract_features_from_file(file_path: str) -> list[np.ndarray]:
    """
    Full pipeline: load → preprocess → segment → extract features.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    list[np.ndarray]
        One feature vector per segment.
    """
    signal = load_and_preprocess(file_path)
    segments = split_into_segments(signal)

    # If the audio is shorter than SEGMENT_DURATION, use the whole thing
    if not segments:
        segments = [signal]

    return [extract_features(seg) for seg in segments]


def extract_features_for_prediction(file_path: str) -> np.ndarray:
    """
    Extract features for a single prediction (averages across segments).

    Parameters
    ----------
    file_path : str
        Path to the uploaded audio file.

    Returns
    -------
    np.ndarray
        Averaged feature vector (1-D, 113 values).
    """
    segment_features = extract_features_from_file(file_path)
    # Average across all segments for a single prediction vector
    return np.mean(segment_features, axis=0)

"""
predict.py
──────────
Loads a trained model and returns genre predictions for new audio files.
Used by the FastAPI server (main.py) to handle prediction requests.
"""

import numpy as np
import joblib
import librosa

from config import (
    GENRES,
    LABEL_ENCODER_PATH,
    RF_MODEL_PATH,
    RF_SCALER_PATH,
    SVM_MODEL_PATH,
    SVM_SCALER_PATH,
    CNN_MODEL_PATH,
    SAMPLE_RATE,
    HOP_LENGTH,
)
from feature_extraction import extract_features_for_prediction, load_and_preprocess


class GenrePredictor:
    """
    Wraps a trained model and scaler to predict music genre from audio files.

    Parameters
    ----------
    model_type : str
        One of 'rf' (Random Forest), 'svm', or 'cnn'.
    """

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._loaded = False

    def load(self) -> None:
        """Load the model, scaler, and label encoder from disk."""
        if self._loaded:
            return

        # Label encoder (shared across all model types)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

        if self.model_type == "rf":
            self.model = joblib.load(RF_MODEL_PATH)
            self.scaler = joblib.load(RF_SCALER_PATH)

        elif self.model_type == "svm":
            self.model = joblib.load(SVM_MODEL_PATH)
            self.scaler = joblib.load(SVM_SCALER_PATH)

        elif self.model_type == "cnn":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
            # CNN uses its own normalization; scaler not needed
            self.scaler = None

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._loaded = True

    def predict(self, file_path: str) -> dict:
        """
        Predict the genre of an audio file.

        Parameters
        ----------
        file_path : str
            Path to the audio file (WAV or MP3).

        Returns
        -------
        dict
            {
                "genre": str,
                "confidence": float,
                "scores": [{"genre": str, "score": float}, ...],
                "features": {
                    "tempo": float,
                    "spectral_centroid": float,
                    "spectral_rolloff": float,
                    "spectral_bandwidth": float,
                    "zero_crossing_rate": float,
                    "energy": float,
                }
            }
        """
        self.load()

        # ── Extract features ──────────────────────────────────────────────
        feature_vector = extract_features_for_prediction(file_path)

        # ── Compute display-friendly audio features ──────────────────────
        signal = load_and_preprocess(file_path)
        audio_features = self._compute_display_features(signal)

        # ── Scale features ────────────────────────────────────────────────
        if self.scaler is not None:
            feature_vector_scaled = self.scaler.transform(
                feature_vector.reshape(1, -1)
            )
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)

        # ── Predict ───────────────────────────────────────────────────────
        if self.model_type == "cnn":
            # CNN expects (samples, features, 1)
            X_cnn = feature_vector_scaled[..., np.newaxis]
            probabilities = self.model.predict(X_cnn, verbose=0)[0]
            predicted_idx = int(np.argmax(probabilities))
            genre = self.label_encoder.inverse_transform([predicted_idx])[0]
        else:
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            predicted_idx = int(np.argmax(probabilities))
            genre = self.model.classes_[predicted_idx]

        confidence = float(probabilities[predicted_idx])

        # ── Build scores list (all genres sorted by probability) ─────────
        if self.model_type == "cnn":
            classes = self.label_encoder.classes_
        else:
            classes = self.model.classes_

        scores = [
            {"genre": cls.capitalize(), "score": round(float(prob), 4)}
            for cls, prob in zip(classes, probabilities)
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "genre": genre.capitalize(),
            "confidence": round(confidence, 4),
            "scores": scores,
            "features": audio_features,
        }

    @staticmethod
    def _compute_display_features(signal: np.ndarray) -> dict:
        """
        Compute human-readable audio features for the UI display.
        These are separate from the ML feature vector.
        """
        sr = SAMPLE_RATE

        # Energy (RMS)
        rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)
        energy = float(np.mean(rms))

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, hop_length=HOP_LENGTH)
        spectral_centroid = float(np.mean(centroid))

        # Spectral roll-off
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, hop_length=HOP_LENGTH)
        spectral_rolloff = float(np.mean(rolloff))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, hop_length=HOP_LENGTH)
        spectral_bandwidth = float(np.mean(bandwidth))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=HOP_LENGTH)
        zero_crossing_rate = float(np.mean(zcr))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr, hop_length=HOP_LENGTH)
        tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)

        return {
            "tempo": round(tempo_val),
            "spectral_centroid": round(spectral_centroid, 2),
            "spectral_rolloff": round(spectral_rolloff, 2),
            "spectral_bandwidth": round(spectral_bandwidth, 2),
            "zero_crossing_rate": round(zero_crossing_rate, 5),
            "energy": round(energy, 4),
        }

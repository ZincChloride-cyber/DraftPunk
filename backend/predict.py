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
    get_prediction_model_info,
    get_trained_models,
)
from feature_extraction import (
    extract_features_for_prediction,
    extract_mel_spectrogram_for_display,
    extract_waveform_for_display,
    load_and_preprocess,
)


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

        # ── Extract features (one vector per 10s segment, GTZAN-style) ───
        segment_features = extract_features_for_prediction(file_path)
        if not segment_features:
            raise ValueError("Could not extract features from audio file.")

        # ── Compute display-friendly audio features ──────────────────────
        signal = load_and_preprocess(file_path, max_duration=None)
        audio_features = self._compute_display_features(signal)
        mel_spectrogram = extract_mel_spectrogram_for_display(signal)
        waveform = extract_waveform_for_display(signal)

        # ── Predict per segment, then average probabilities ─────────────
        # Training uses each segment as its own sample; averaging raw features
        # before predict does not match what the model learned.
        prob_list: list[np.ndarray] = []
        for feature_vector in segment_features:
            if self.scaler is not None:
                X = self.scaler.transform(feature_vector.reshape(1, -1))
            else:
                X = feature_vector.reshape(1, -1)

            if self.model_type == "cnn":
                X_cnn = X[..., np.newaxis]
                prob_list.append(self.model.predict(X_cnn, verbose=0)[0])
            else:
                prob_list.append(self.model.predict_proba(X)[0])

        probabilities = np.mean(prob_list, axis=0)
        predicted_idx = int(np.argmax(probabilities))

        if self.model_type == "cnn":
            genre = self.label_encoder.inverse_transform([predicted_idx])[0]
        else:
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
            "mel_spectrogram": mel_spectrogram,
            "waveform": waveform,
            "models": {
                "prediction": get_prediction_model_info(),
                "trained": get_trained_models(),
                "feature_extraction": "librosa (MFCC, chroma, mel-spectrogram, spectral)",
                "dataset": "GTZAN",
            },
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

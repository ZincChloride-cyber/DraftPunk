"""
config.py
─────────
Central configuration for the Draft Punk ML backend.
All paths, constants, and genre labels live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Model file paths ──────────────────────────────────────────────────────────
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
RF_SCALER_PATH = MODELS_DIR / "rf_scaler.joblib"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"
SVM_SCALER_PATH = MODELS_DIR / "svm_scaler.joblib"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE = 22050              # Hz — standard for music analysis
AUDIO_DURATION = 30              # seconds — GTZAN track length
SEGMENT_DURATION = 10            # seconds — split each track into segments
N_MFCC = 40                     # number of MFCC coefficients
N_CHROMA = 12                   # chroma feature bins (one per note C–B)
HOP_LENGTH = 512                 # samples between frames
N_FFT = 2048                    # FFT window size
SILENCE_THRESHOLD = -60          # dB — trim silence below this

# ── Genre labels (GTZAN) ─────────────────────────────────────────────────────
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# ── Training ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
RF_N_ESTIMATORS = 200            # number of trees in Random Forest
SVM_KERNEL = "rbf"               # SVM kernel type
CV_FOLDS = 5                     # stratified k-fold cross-validation

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:8081",
    "http://localhost:8082",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:8082",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

# ── ML model metadata (for API / UI) ─────────────────────────────────────────
MODEL_LABELS: dict[str, dict[str, str]] = {
    "rf": {
        "name": "Random Forest",
        "framework": "scikit-learn",
        "description": "Ensemble of decision trees on MFCC & spectral features",
    },
    "svm": {
        "name": "Support Vector Machine",
        "framework": "scikit-learn",
        "description": "RBF-kernel SVM on scaled acoustic features",
    },
    "cnn": {
        "name": "Convolutional Neural Network",
        "framework": "TensorFlow / Keras",
        "description": "Deep CNN on segment feature vectors",
    },
}

MODEL_PATHS: dict[str, Path] = {
    "rf": RF_MODEL_PATH,
    "svm": SVM_MODEL_PATH,
    "cnn": CNN_MODEL_PATH,
}

# Model used by the live API for /api/predict
ACTIVE_PREDICTION_MODEL = "rf"


def get_trained_models() -> list[dict[str, str]]:
    """Return metadata for each trained model file present on disk."""
    trained: list[dict[str, str]] = []
    for model_id, path in MODEL_PATHS.items():
        if path.exists():
            trained.append({"id": model_id, **MODEL_LABELS[model_id]})
    return trained


def get_prediction_model_info() -> dict[str, str]:
    """Metadata for the model that serves live predictions."""
    return {
        "id": ACTIVE_PREDICTION_MODEL,
        **MODEL_LABELS[ACTIVE_PREDICTION_MODEL],
    }


# ── GTZAN dataset URL (Kaggle mirror) ────────────────────────────────────────
GTZAN_DOWNLOAD_URL = os.environ.get(
    "GTZAN_URL",
    "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification",
)

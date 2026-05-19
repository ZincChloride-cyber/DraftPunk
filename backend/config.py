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

# ── GTZAN dataset URL (Kaggle mirror) ────────────────────────────────────────
GTZAN_DOWNLOAD_URL = os.environ.get(
    "GTZAN_URL",
    "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification",
)

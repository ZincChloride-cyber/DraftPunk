"""
main.py
───────
FastAPI server for the Draft Punk genre classification API.

Endpoints
─────────
  POST /api/predict   — Upload an audio file → returns genre prediction
  GET  /api/health    — Health check / readiness probe

Start with:
    cd backend
    python -m uvicorn main:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    RF_MODEL_PATH,
    get_prediction_model_info,
    get_trained_models,
)
from predict import GenrePredictor

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Draft Punk — Genre Classification API",
    description="Upload a song and get its predicted genre using ML.",
    version="1.0.0",
)

# CORS — allow the Vite frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load predictor (lazy — loads model on first request) ─────────────────────
predictor = GenrePredictor(model_type="rf")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Helpful landing page when visiting the API base URL in a browser."""
    return {
        "name": "Draft Punk — Genre Classification API",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "POST /api/predict",
    }


@app.get("/api/health")
async def health_check():
    """Check if the API is running and the model is available."""
    model_available = RF_MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": predictor._loaded,
        "model_available": model_available,
        "models": {
            "prediction": get_prediction_model_info(),
            "trained": get_trained_models(),
        },
    }


@app.post("/api/predict")
async def predict_genre(file: UploadFile = File(...)):
    """
    Accept an audio file upload and return the predicted genre.

    Parameters
    ----------
    file : UploadFile
        An MP3 or WAV audio file.

    Returns
    -------
    JSON response with genre, confidence, scores, and audio features.
    """
    # ── Validate file type ────────────────────────────────────────────────
    allowed_types = {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
        "audio/x-wav", "audio/x-m4a", "audio/mp4",
    }
    allowed_extensions = {".mp3", ".wav", ".wave", ".m4a"}

    file_ext = Path(file.filename or "").suffix.lower()

    if file.content_type not in allowed_types and file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Please upload an MP3 or WAV file.",
        )

    # ── Check if model is available ───────────────────────────────────────
    if not RF_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run 'python train_model.py' first.",
        )

    # ── Save upload to a temp file ────────────────────────────────────────
    try:
        suffix = file_ext if file_ext else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # ── Run prediction ────────────────────────────────────────────────────
    try:
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEV SERVER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)

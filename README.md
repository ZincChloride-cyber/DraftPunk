---
title: Draftpunk Api
emoji: 🦀
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
---

# Draft Punk

Draft Punk is a full-stack web application designed for music genre classification. It uses machine learning to analyze audio files and predict their musical genre with high accuracy. 

The application is split into a modern React frontend and a powerful Python FastAPI backend that serves trained Machine Learning models (Random Forest, SVM, CNN) trained on the GTZAN dataset.

##  Features

- **Upload & Analyze**: Upload `.mp3` or `.wav` files directly from the UI.
- **Machine Learning Powered**: Uses Random Forest (primary), SVM, or CNN models for prediction.
- **Feature Extraction**: Extracts acoustic features using `librosa` (MFCCs, Chroma, Mel-spectrograms).
- **Modern UI**: Built with React, Vite, Tailwind CSS, and shadcn/ui components for a premium user experience.
- **Fast API**: High-performance backend built with FastAPI.

##  Technology Stack

**Frontend:**
- React 19
- Vite
- TanStack Router & Query
- Tailwind CSS
- shadcn/ui (Radix UI)
- Lucide React (Icons)
- Recharts (Data visualization)
- Three.js & Vanta.js (Background animations)

**Backend:**
- Python 3.9+
- FastAPI
- scikit-learn (Machine Learning Models)
- TensorFlow / Keras (CNN Model)
- Librosa (Audio Processing)
- Pandas & NumPy
- Uvicorn

##  Project Structure

```text
Draft Punk/
├── backend/                  # Python FastAPI Backend & ML Pipeline
│   ├── data/                 # GTZAN dataset (downloaded during training)
│   ├── models/               # Saved trained ML models (.joblib, .keras)
│   ├── config.py             # Configuration and constants
│   ├── feature_extraction.py # Audio feature extraction logic
│   ├── main.py               # FastAPI server and endpoints
│   ├── predict.py            # Prediction pipeline
│   ├── train_model.py        # Script to train models on GTZAN
│   └── requirements.txt      # Python dependencies
├── src/                      # React Frontend Source Code
├── package.json              # NPM dependencies and scripts
└── vite.config.ts            # Vite configuration
```

##  Initialization & Setup

To run Draft Punk locally, you need to set up both the backend and the frontend.

### 1. Backend Setup (Machine Learning & API)

First, navigate to the backend directory and set up a Python virtual environment.

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Train the Models:**
Before running the API, you need to train the machine learning models. The script will automatically attempt to download the GTZAN dataset from Kaggle.

```bash
python train_model.py
```
*(Optional: Use `python train_model.py --skip-cnn` to skip CNN training if TensorFlow/GPU is not set up).*

**Start the API Server:**
Once the models are trained and saved in the `models/` directory, start the FastAPI server:

```bash
python -m uvicorn main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`. You can visit `http://localhost:8000/docs` for the interactive Swagger API documentation.

### 2. Frontend Setup (React UI)

Open a new terminal window, ensuring you are in the root directory of the project (`Draft Punk/`).

```bash
# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```
The frontend will be available at `http://localhost:5173`.

## API Endpoints

- `GET /api/health` - Check if the API is running and the models are loaded.
- `POST /api/predict` - Upload an audio file (multipart/form-data) to get a genre prediction, confidence scores, and audio feature data.

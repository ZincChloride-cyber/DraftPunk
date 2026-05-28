FROM python:3.11-slim

# Install system audio libraries required by Librosa and soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements and install python packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all backend source code and models
COPY . /code

# Expose port 7860 (Hugging Face default) and run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

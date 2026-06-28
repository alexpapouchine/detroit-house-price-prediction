# Detroit House Price Prediction API — container image
# Built for deployment to Google Cloud Run

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by xgboost / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-trained model artifacts
COPY main.py .
COPY model/ ./model/

# Cloud Run injects the PORT environment variable; default to 8080 locally
ENV PORT=8080
ENV API_KEY=dev-detroit-housing-key-2026
EXPOSE 8080

# Run with uvicorn; host 0.0.0.0 is required for Cloud Run to route traffic
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

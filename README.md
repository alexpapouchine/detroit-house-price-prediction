# Detroit House Price Prediction API

A machine learning model that predicts residential sale prices in Detroit, MI, deployed as a containerized REST API on **Google Cloud Platform (Cloud Run)**.

Built for: *Machine Learning Model Deployment Assignment - Spring 2026*

---

## Project Structure

```
.
├── notebook/
│   ├── Detroit_House_Price_Model.ipynb   # EDA, preprocessing, training, evaluation
│   └── detroit_housing.csv               # Dataset
├── generate_data.py                      # Script that generated the dataset
├── model/
│   ├── detroit_house_price_model.joblib  # Trained pipeline (preprocessing + XGBoost)
│   └── feature_schema.json               # Feature schema used for input validation
├── app/
│   ├── main.py                           # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── model/                            # Copy of model artifacts used by the container
└── README.md
```

## Model

- **Problem:** Regression — predict `sale_price` from 19 structural, location, and condition features.
- **Models trained:** Ridge Regression, Random Forest, XGBoost (tuned via `GridSearchCV`).
- **Selected model:** Tuned XGBoost — Test RMSE ≈ $27,974, R² ≈ 0.945.
- **Preprocessing:** `ColumnTransformer` with median/most-frequent imputation, `StandardScaler`, and `OneHotEncoder`, saved as part of the deployed pipeline so the exact same transformations are applied at inference time.

See `notebook/Detroit_House_Price_Model.ipynb` for the full EDA, training, and evaluation walkthrough.

## Running locally (without Docker)

```bash
cd app
pip install -r requirements.txt
export API_KEY="your-chosen-key"
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Visit `http://localhost:8080/docs` for interactive Swagger documentation.

## Running with Docker

```bash
cd app
docker build -t detroit-house-price-api .
docker run -p 8080:8080 -e API_KEY="your-chosen-key" detroit-house-price-api
```

## API Usage

### Health check
```bash
curl http://localhost:8080/health
```

### Predict
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-chosen-key" \
  -d '{
    "neighborhood": "Midtown",
    "sqft": 1450,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "lot_size_sqft": 4200,
    "year_built": 1965,
    "stories": 2,
    "garage_spaces": 1,
    "has_basement": 1,
    "has_porch": 1,
    "renovated_last_10yrs": 1,
    "fireplace": 0,
    "condition_score": 7.2,
    "distance_to_downtown_km": 2.5,
    "crime_index": 35,
    "school_rating": 6.5,
    "walk_score": 78,
    "property_tax_annual": 2100,
    "vacant_lot_nearby": 0
  }'
```

Response:
```json
{
  "predicted_sale_price": 255983.34,
  "currency": "USD",
  "model_version": "xgboost-v1.0"
}
```

Valid `neighborhood` values are listed in `model/feature_schema.json`.

## Deploying to Google Cloud Platform (Cloud Run)

```bash
# 1. Authenticate and set your project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# 3. Build and push the container image using Cloud Build
cd app
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/detroit-house-price-api

# 4. Deploy to Cloud Run
gcloud run deploy detroit-house-price-api \
  --image gcr.io/YOUR_PROJECT_ID/detroit-house-price-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_KEY="your-chosen-key" \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 3

# 5. Test the deployed endpoint
curl -X POST https://YOUR-SERVICE-URL/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-chosen-key" \
  -d '{ ... }'
```

`--min-instances 0` allows the service to scale to zero when idle (cost control on free tier); `--max-instances 3` caps scale-out. Cloud Run autoscales between these bounds based on concurrent request load, satisfying the "scalable and resilient" deployment requirement.

## Security Notes

- The `/predict` endpoint requires a valid `x-api-key` header.
- For production use, replace the environment-variable API key with a secret stored in **Google Secret Manager** and mounted at runtime.
- This assignment intentionally uses a single static key for simplicity; a production system would use per-client keys with rotation and rate limiting.

## Restrictions Honored

- No GUI tool (Streamlit/Gradio) is used for the core API - FastAPI serves a direct REST API only.
- The model is trained once in the notebook and saved with `joblib`; the API loads it once at startup (`joblib.load` at module import time) and **never calls `.fit()`** at request time.

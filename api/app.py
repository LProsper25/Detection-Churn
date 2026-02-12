import joblib
import pandas as pd
from fastapi import FastAPI
from src.preprocess import clean_data
from src.decision import predict_with_threshold
from api.schema import Customer
from src.config import MODEL_PATH

app = FastAPI(
    title="Churn Detection API",
    description="API for detecting customer churn using a pre-trained model.",
    version="1.0.0"
)

# Charger le model
pipeline = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "message": "Churn detection API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(customer: Customer):
    # Convertir data en DataFrame
    df = pd.DataFrame([customer.dict()])

    # Nettoyage
    X = clean_data(df)

    # Probabilité
    proba = pipeline.predict_proba(X)[:, 1]

    # Prédiction
    pred = predict_with_threshold(pipeline, X)
    
    return {
        "Churn_probability": float(proba[0]),
        "Prediction": int(pred[0])
    }
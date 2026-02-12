import pandas as pd
import joblib
from decision import predict_with_threshold
from preprocess import clean_data
from config import MODEL_PATH

# Modèle
pipeline = joblib.load(MODEL_PATH)

# Dataset
sample = pd.DataFrame([{
    "credit_score": 500,
    "country": "germany",
    "gender": "Male",
    "age": 60,
    "tenure": 3,
    "balance": 0.0,
    "products_number": 1,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 50000
}])

# Nettoyage
X = clean_data(sample)

# Prédiction et Probabilité
pred = predict_with_threshold(pipeline, X)
proba = pipeline.predict_proba(X)[:, 1]
print(pred, proba)
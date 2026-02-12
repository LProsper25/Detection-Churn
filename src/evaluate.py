from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from utils import save_metrics
from decision import predict_with_threshold
from config import MODEL_PATH, X_TEST, Y_TEST
import joblib
import numpy as np
import pandas as pd

# Split
X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).squeeze()

# Chargement du modèle
pipeline = joblib.load(MODEL_PATH)

# Prédiction
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

metrics = {
    'ROC-AUC Score': roc_auc,
    'F1 Score': f1
}
save_metrics(metrics)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print()

# Seuil de décision
thresholds = np.arange(0.2, 0.8, 0.02)
best_thresh = 0
best_f1 = 0

for t in thresholds:
    preds = (y_proba > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best threshold:", best_thresh)
print("Best F1:", best_f1)
print()

pred = predict_with_threshold(pipeline, X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
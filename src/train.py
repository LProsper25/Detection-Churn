import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from preprocess import create_preprocessor
from config import RANDOM_STATE, N_SPLITS, X_TRAIN, X_TEST, Y_TEST, Y_TRAIN

# Split
X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze()
X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).squeeze()

# Preprocessing
preprocessor = create_preprocessor(X_train)

# Tester plusieurs modèles
model1 = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))
    ]
)
model2 = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', SVC(class_weight='balanced', kernel='rbf', random_state=RANDOM_STATE))
    ]
)
model3 = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, class_weight='balanced', scale_pos_weight=3, random_state=RANDOM_STATE))
    ]
)
model4 = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', RandomForestClassifier(n_estimators=300, max_depth=3, class_weight='balanced', random_state=RANDOM_STATE))
    ]
)


list_model = {"logistic": model1, "svm": model2, 'xgb': model3, "random": model4}

for names, model in list_model.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(names, classification_report(y_test, y_pred))



# Modèle choisi
model = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, scale_pos_weight=3, random_state=RANDOM_STATE)
pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

# Cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"ROC-AUC CV Scores: {scores}")
print(f"MEAN ROC-AUC: {np.mean(scores)}")

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Sauvegarde du pipeline
joblib.dump(pipeline, "model/churn_pipeline.pkl")
print("Modèle sauvegardé")
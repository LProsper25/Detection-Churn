import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from preprocess import create_preprocessor
from config import X_TEST, X_TRAIN, MODEL_PATH, Y_TRAIN

# Split
X_test = pd.read_csv(X_TEST)
X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN)

# Chargement du modèle
model = joblib.load(MODEL_PATH)

# Créer et fit le preprocessor pour éviter les fuites de données
preprocessor = create_preprocessor(X_train)
preprocessor.fit(X_train, y_train)

# SHAP
explainer = shap.TreeExplainer(model.named_steps['model'])
X_test_preprocessed = preprocessor.transform(X_test)
shap_values = explainer(X_test_preprocessed)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out())
plt.show()
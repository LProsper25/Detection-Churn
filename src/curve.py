import joblib
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import MODEL_PATH, X_TEST, X_TRAIN, Y_TEST, Y_TRAIN, N_SPLITS, RANDOM_STATE

# Split
X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze()
X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).squeeze()

# Modèle
pipeline = joblib.load(MODEL_PATH)

def learning_curves():
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_size, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='roc_auc',
        cv=cv
    )
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.plot(train_size, train_scores.mean(axis=1), label='train')
    plt.plot(train_size, val_scores.mean(axis=1), label='validation')
    plt.legend()
    plt.show()


def preci_recall_curves():
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, threshold = precision_recall_curve(y_test, y_proba)
    plt.plot(threshold, precision[:-1], label='precision')
    plt.plot(threshold, recall[:-1], label='recall')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    learning_curves()
    preci_recall_curves()
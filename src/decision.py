# Décision finale

def predict_with_threshold(pipeline, X, threshold=0.40):
    y_proba = pipeline.predict_proba(X)[:, 1]
    pred = (y_proba >= threshold).astype(int)
    return pred
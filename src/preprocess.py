import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Chargement du dataset
def load_data(path):
    df = pd.read_csv(path)
    return df


# Néttoyage du data & Features engineering
def clean_data(df):
    # Suppression des doublons
    df = df.drop_duplicates()

    # Feature engineering
    df["many_products"] = (df["products_number"] >= 2).astype(int)
    df["low_credit_score"] = (df["credit_score"] < 600).astype(int)
    df["is senior"] = (df["age"] >= 50).astype(int)
    df["high_balance"] = (df["balance"] > df["balance"].median()).astype(int)
    df["balance_salary_ratio"] = (df["balance"] / df["estimated_salary"] + 1)

    # Columns delete
    df = df.drop(columns=["customer_id"], errors='ignore')

    return df


# Séparation de features
def feature_target(df):
    # Séparation X, y
    y = df["churn"]
    X = df.drop(columns="churn", axis=1)

    return X, y


# Préprocessing
def create_preprocessor(X):
    # Selection des features
    num_features = X.select_dtypes(include=['float', 'int']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    # Scaling
    num_scaler = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    # Encoding
    cat_encoder = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_scaler, num_features),
            ('cat', cat_encoder, cat_features)
        ]
    )
    return preprocessor
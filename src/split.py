from sklearn.model_selection import train_test_split
from preprocess import load_data, clean_data, feature_target
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def main():
    # Charger le dataset
    data = load_data(path=DATA_PATH)

    # Nettoyer le dataset & features engineering
    df = clean_data(data)
    X, y = feature_target(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Sauvegarde
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(df.shape)
    print("Splits train/test créés et sauvegardés.")

if __name__ == "__main__":
    main()
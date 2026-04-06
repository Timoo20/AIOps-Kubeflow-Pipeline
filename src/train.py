import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

DATA_PATH = "data/processed.csv"
MODEL_DIR = "model"
MODEL_PATH = "model/model.joblib"

def train():
    print("[INFO] Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])

    print("[INFO] Training Isolation Forest model...")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    model.fit(X)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("[SUCCESS] Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train()

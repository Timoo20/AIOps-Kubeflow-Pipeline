import pandas as pd
import joblib
from sklearn.metrics import classification_report

DATA_PATH = "data/processed.csv"
MODEL_PATH = "model/model.joblib"

def evaluate():
    print("[INFO] Loading data and model...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    model = joblib.load(MODEL_PATH)

    print("[INFO] Running predictions...")

    preds = model.predict(X)

    # Convert IsolationForest output
    preds = [1 if p == -1 else 0 for p in preds]

    print("[INFO] Evaluation Report:\n")
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate()

import pandas as pd
import os

INPUT_PATH = "data/creditcard.csv"
OUTPUT_PATH = "data/processed.csv"

def preprocess():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("[INFO] Cleaning data...")

    # Normalize Amount
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # Drop Time column
    df.drop(columns=['Time'], inplace=True)

    print("[INFO] Saving processed data...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("[SUCCESS] Preprocessing completed!")

if __name__ == "__main__":
    preprocess()

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud"
    output_dir = "data"

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset, path=output_dir, unzip=True)

    print("[SUCCESS] Dataset downloaded and extracted!")

if __name__ == "__main__":
    download_dataset()

# Kubeflow End-to-End MLOps Lab 

## AIOps Anomaly Detection Pipeline using Kubeflow on Kubernetes

---

# Project Structure 

```
kubeflow-aiops-lab/
│
├── README.md
├── requirements.txt
├── .gitignore
├── docker/
│   ├── Dockerfile.preprocessing
│   ├── Dockerfile.training
│   ├── Dockerfile.evaluation
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── pipelines/
│   ├── pipeline.py
│   └── compiled_pipeline.yaml
│
├── data/
│   └── (auto-fetched from Kaggle)
│
├── scripts/
│   ├── download_data.py
│   └── build_images.sh
│
└── k8s/
    └── deployment.yaml
```

---

# 2. Requirements

```
kubeflow-pipelines
pandas
numpy
scikit-learn
kaggle
joblib
```

---

# 3. Kaggle Data Fetch Script

## scripts/download_data.py

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

DATASET = "mlg-ulb/creditcardfraud"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
api.dataset_download_files(DATASET, path=OUTPUT_DIR, unzip=True)

print("Dataset downloaded successfully")
```

---

# 4. Data Preprocessing

## src/preprocessing.py

```python
import pandas as pd
import joblib

INPUT_PATH = "/data/creditcard.csv"
OUTPUT_PATH = "/data/processed.csv"


def preprocess():
    df = pd.read_csv(INPUT_PATH)

    # Normalize Amount
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # Drop Time column
    df = df.drop(columns=['Time'])

    df.to_csv(OUTPUT_PATH, index=False)
    print("Preprocessing complete")


if __name__ == "__main__":
    preprocess()
```

---

# 5. Model Training

## src/train.py

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

DATA_PATH = "/data/processed.csv"
MODEL_PATH = "/model/model.joblib"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Class'])

    model = IsolationForest(n_estimators=100, contamination=0.01)
    model.fit(X)

    joblib.dump(model, MODEL_PATH)
    print("Model training complete")


if __name__ == "__main__":
    train()
```

---

# 6. Model Evaluation

## src/evaluate.py

```python
import pandas as pd
import joblib
from sklearn.metrics import classification_report

DATA_PATH = "/data/processed.csv"
MODEL_PATH = "/model/model.joblib"


def evaluate():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Class'])
    y = df['Class']

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    preds = [1 if p == -1 else 0 for p in preds]

    print(classification_report(y, preds))


if __name__ == "__main__":
    evaluate()
```

---

# 7. Dockerfiles

## docker/Dockerfile.training

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["python", "src/train.py"]
```

(Repeat similarly for preprocessing & evaluation)

---

# 8. Kubeflow Pipeline

## pipelines/pipeline.py

```python
import kfp
from kfp import dsl

@dsl.component
def preprocess_op():
    return

@dsl.component
def train_op():
    return

@dsl.component
def eval_op():
    return

@dsl.pipeline(name="aiops-anomaly-detection")
def pipeline():
    preprocess = preprocess_op()
    train = train_op()
    eval = eval_op()

    train.after(preprocess)
    eval.after(train)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "compiled_pipeline.yaml")
```

---

# 9. Build & Push Docker Images

## scripts/build_images.sh

```bash
docker build -t preprocessing -f docker/Dockerfile.preprocessing .
docker build -t training -f docker/Dockerfile.training .
docker build -t evaluation -f docker/Dockerfile.evaluation .
```

---
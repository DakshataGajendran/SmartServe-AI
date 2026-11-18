import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate() -> None:
    # 1. Load processed splits
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df["ticket_text"], train_df["category"]
    X_test, y_test = test_df["ticket_text"], test_df["category"]

    # 2. Layer 4: TF-IDF (fit ONLY on training data to avoid data leakage)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 3. Layer 5: Logistic Regression Classifier
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    # 4. Layer 6 & 7: Prediction, Probabilities & Latency Benchmark
    start_time = time.perf_counter()
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)
    total_ms = (time.perf_counter() - start_time) * 1000
    avg_latency_ms = total_ms / len(X_test)

    # 5. Layer 17: Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("=== MODEL 1: TF-IDF + LOGISTIC REGRESSION ===")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Avg Latency/Ticket: {avg_latency_ms:.4f} ms")
    print(f"Mean Max Conf     : {np.max(y_proba, axis=1).mean():.4f}\n")

    print("=== PER-CLASS REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== CONFUSION MATRIX (Rows=Actual, Cols=Predicted) ===")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # 6. Save artifacts & benchmark metrics
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(model, MODELS_DIR / "logreg_model.joblib")

    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "model_version": "v1.0-logreg",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "avg_latency_ms": round(float(avg_latency_ms), 4),
        "classes": labels,
    }
    with open(MODELS_DIR / "logreg_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts to models/: tfidf_vectorizer.joblib, logreg_model.joblib, logreg_metrics.json")


if __name__ == "__main__":
    train_and_evaluate()
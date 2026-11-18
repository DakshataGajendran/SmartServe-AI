import json
import time
from pathlib import Path

import joblib
import numpy as np

from app.ml.preprocess import clean_ticket_text
from app.rules.triage_rules import evaluate_review_and_routing

MODELS_DIR = Path("models")


class SmartServePredictor:
    def __init__(self) -> None:
        self.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        self.model = joblib.load(MODELS_DIR / "logreg_model.joblib")
        with open(MODELS_DIR / "logreg_metrics.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_version = meta.get("model_version", "v1.0-logreg")

    def predict(self, raw_text: str) -> dict:
        start = time.perf_counter()
        cleaned = clean_ticket_text(raw_text)

        if not cleaned:
            raise ValueError("Ticket text cannot be empty or whitespace only.")

        features = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))

        category = str(self.model.classes_[best_idx])
        confidence = round(float(probs[best_idx]), 4)

        routing_info = evaluate_review_and_routing(cleaned, category, confidence)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        return {
            "category": category,
            "confidence": confidence,
            "priority": routing_info["priority"],
            "priority_reason": routing_info["priority_reason"],
            "suggested_team": routing_info["suggested_team"],
            "confidence_band": routing_info["confidence_band"],
            "human_review_required": routing_info["human_review_required"],
            "review_reason": routing_info["review_reason"],
            "latency_ms": latency_ms,
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    predictor = SmartServePredictor()
    samples = [
        "Please order a new ergonomic keyboard and mouse for new hire",
        "Not working",
        "Someone appears to be accessing my account without authorization",
        "Laptop overheating, Wi-Fi down and Outlook broken",
    ]
    for s in samples:
        print(f"\nTicket: {s}")
        print(predictor.predict(s))
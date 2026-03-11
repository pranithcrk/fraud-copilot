"""ML fraud scoring module using trained model."""

import os
import pickle
import numpy as np


class FraudScorer:
    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model at {self.model_path} — using random scoring for demo")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray) -> tuple[float, list[dict]]:
        if self.model is not None:
            prob = self.model.predict_proba(features)[0][1]
            importances = self.model.feature_importances_
            feature_names = [
                "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
                "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
                "V28", "Amount",
            ]
            top_idx = np.argsort(importances)[-5:][::-1]
            top_features = [
                {"name": feature_names[i], "importance": round(float(importances[i]), 4), "value": round(float(features[0][i]), 4)}
                for i in top_idx
            ]
        else:
            # Demo mode: heuristic scoring
            amount = features[0][-1]
            v14 = features[0][14] if features.shape[1] > 14 else 0
            prob = min(1.0, max(0.0, (amount / 10000) + abs(v14) / 20))
            top_features = [
                {"name": "Amount", "importance": 0.45, "value": round(float(amount), 2)},
                {"name": "V14", "importance": 0.22, "value": round(float(v14), 4)},
                {"name": "V3", "importance": 0.11, "value": round(float(features[0][3]), 4)},
            ]

        return float(prob), top_features

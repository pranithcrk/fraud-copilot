"""FastAPI backend for Fraud Intelligence Copilot."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from .scorer import FraudScorer
from .llm import FraudExplainer

app = FastAPI(title="Fraud Intelligence Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = FraudScorer()
explainer = FraudExplainer()


class Transaction(BaseModel):
    amount: float
    time: float
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": scorer.is_loaded()}


@app.post("/analyze")
async def analyze_transaction(txn: Transaction):
    features = np.array([[
        txn.time, txn.v1, txn.v2, txn.v3, txn.v4, txn.v5, txn.v6, txn.v7,
        txn.v8, txn.v9, txn.v10, txn.v11, txn.v12, txn.v13, txn.v14,
        txn.v15, txn.v16, txn.v17, txn.v18, txn.v19, txn.v20, txn.v21,
        txn.v22, txn.v23, txn.v24, txn.v25, txn.v26, txn.v27, txn.v28,
        txn.amount,
    ]])

    score, top_features = scorer.predict(features)
    explanation = await explainer.explain(txn.amount, score, top_features)

    return {
        "fraud_score": round(score, 4),
        "risk_level": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW",
        "top_features": top_features,
        "explanation": explanation,
    }


# Sample transactions for demo
SAMPLES = [
    {"label": "Normal purchase", "amount": 42.50, "time": 50000, "v1": -1.36, "v2": -0.07, "v3": 2.54, "v14": -0.31},
    {"label": "Suspicious large", "amount": 9999.99, "time": 3600, "v1": -5.41, "v2": 3.88, "v3": -7.03, "v14": -12.89},
    {"label": "Micro fraud", "amount": 1.00, "time": 80000, "v1": -3.04, "v2": -3.15, "v3": 1.09, "v14": -8.56},
]


@app.get("/samples")
def get_samples():
    return SAMPLES

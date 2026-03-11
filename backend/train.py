"""Train fraud detection model on Kaggle credit card fraud dataset."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train(data_path: str, output_path: str = "models/fraud_model.pkl"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset: {len(df)} rows, {df['Class'].sum()} fraud cases ({df['Class'].mean():.4%})")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training GradientBoosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Results ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", default="models/fraud_model.pkl")
    args = parser.parse_args()
    train(args.data, args.output)

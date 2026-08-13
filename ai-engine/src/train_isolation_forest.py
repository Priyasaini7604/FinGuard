"""
FinGuard - Isolation Forest Training
Trains an unsupervised anomaly detector on transaction behavioral features.
Isolation Forest works by randomly splitting data - anomalies get isolated
in fewer splits than normal points, since they don't blend into the crowd.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Features that describe BEHAVIOR, not identity - we deliberately exclude
# account IDs, transaction IDs, and timestamps since the model should learn
# patterns, not memorize specific accounts.
FEATURE_COLS = [
    "amount",
    "txn_count_1h",
    "unique_receivers_1h",
    "total_sent_1h",
    "avg_amount_sender",
    "amount_deviation",
    "near_threshold_flag",
    "in_out_ratio",
]


def load_features(path="data/features.csv"):
    df = pd.read_csv(path)
    return df


def train_isolation_forest(X_train):
    """
    contamination = expected proportion of anomalies in the data.
    We set it close to our known fraud ratio (~3%) so the model calibrates
    its decision boundary sensibly. In a real production system without
    labels, you'd estimate this from domain knowledge instead.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def main():
    print("Loading feature data...")
    df = load_features()

    X = df[FEATURE_COLS].fillna(0)
    y_true = df["is_fraud"]

    print("Scaling features (Isolation Forest is distance/split based, scaling helps consistency)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")
    model = train_isolation_forest(X_scaled)

    # decision_function: higher = more normal, lower = more anomalous
    # We flip the sign so higher score = more suspicious (easier to reason about)
    raw_scores = model.decision_function(X_scaled)
    anomaly_score = -raw_scores  # higher = more suspicious
    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    predicted_fraud = (predictions == -1).astype(int)

    df["if_anomaly_score"] = anomaly_score
    df["if_predicted_fraud"] = predicted_fraud

    print("\n--- Evaluation against known fraud labels ---")
    print(classification_report(y_true, predicted_fraud, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, anomaly_score):.4f}")

    # Save model, scaler, and scored data for later use in the ensemble + API
    joblib.dump(model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    df.to_csv("data/features_scored.csv", index=False)

    print("\nSaved model to models/isolation_forest.pkl")
    print("Saved scaler to models/scaler.pkl")
    print("Saved scored data to data/features_scored.csv")


if __name__ == "__main__":
    main()
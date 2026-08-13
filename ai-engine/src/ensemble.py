"""
FinGuard - Ensemble Scoring
Combines Isolation Forest and Autoencoder anomaly scores into a single
unified risk score using min-max normalization + simple averaging.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# Flag as fraud if final ensemble score is above this percentile.
# 97th percentile ~ matches our known ~3% fraud ratio.
FLAG_PERCENTILE = 97


def min_max_normalize(series):
    """
    Scales any score to a 0-1 range so different models' scores
    become comparable before averaging.
    0 = least anomalous in this dataset, 1 = most anomalous.
    """
    return (series - series.min()) / (series.max() - series.min())


def main():
    print("Loading scored outputs from both models...")
    if_scores = pd.read_csv("data/features_scored.csv")
    ae_scores = pd.read_csv("data/features_scored_ae.csv")

    # Both files share the same rows/order (same source features.csv),
    # so we can safely merge on transaction_id to be extra safe.
    merged = if_scores[["transaction_id", "is_fraud", "if_anomaly_score"]].merge(
        ae_scores[["transaction_id", "ae_anomaly_score"]],
        on="transaction_id"
    )

    print("Normalizing both scores to a common 0-1 scale...")
    merged["if_score_norm"] = min_max_normalize(merged["if_anomaly_score"])
    merged["ae_score_norm"] = min_max_normalize(merged["ae_anomaly_score"])

    print("Averaging normalized scores into final ensemble risk score...")
    merged["risk_score"] = (merged["if_score_norm"] + merged["ae_score_norm"]) / 2

    # Flag top ~3% highest risk scores as fraud predictions
    threshold = np.percentile(merged["risk_score"], FLAG_PERCENTILE)
    merged["predicted_fraud"] = (merged["risk_score"] >= threshold).astype(int)

    print("\n--- Ensemble Evaluation against known fraud labels ---")
    print(classification_report(
        merged["is_fraud"], merged["predicted_fraud"],
        target_names=["Normal", "Fraud"]
    ))
    print(f"Ensemble ROC-AUC Score: {roc_auc_score(merged['is_fraud'], merged['risk_score']):.4f}")

    # Save final combined output - this is what the backend will eventually serve
    output_cols = [
        "transaction_id", "is_fraud",
        "if_anomaly_score", "ae_anomaly_score",
        "risk_score", "predicted_fraud"
    ]
    merged[output_cols].to_csv("data/final_risk_scores.csv", index=False)
    print("\nSaved final combined risk scores to data/final_risk_scores.csv")


if __name__ == "__main__":
    main()
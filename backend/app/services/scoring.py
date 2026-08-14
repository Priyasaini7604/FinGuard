"""
FinGuard - ML Scoring Service
Loads both ONNX models once at startup and provides a function to score
a single transaction in real time using recent history from MongoDB.
"""

import onnxruntime as ort
import numpy as np
from datetime import datetime, timedelta, timezone

# Same feature order used during training - must match exactly,
# otherwise the model will misinterpret which number means what.
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

THRESHOLD = 50000
NEAR_THRESHOLD_RATIO = 0.95


class FraudScorer:
    """
    Wraps both ONNX models + their scalers so main.py can load this ONCE
    at startup (in lifespan) and reuse it for every request, instead of
    reloading models from disk on every single transaction.
    """

    def __init__(self, models_dir="models"):
        print("Loading ONNX models into memory...")
        self.if_session = ort.InferenceSession(f"{models_dir}/isolation_forest.onnx")
        self.ae_session = ort.InferenceSession(f"{models_dir}/autoencoder.onnx")

        # Scaler mean_/scale_ arrays saved as plain .npy files during training -
        # loading these needs only numpy, so the backend never needs
        # scikit-learn installed just to unpickle a StandardScaler object.
        self.if_mean = np.load(f"{models_dir}/if_scaler_mean.npy")
        self.if_scale = np.load(f"{models_dir}/if_scaler_scale.npy")
        self.ae_mean = np.load(f"{models_dir}/ae_scaler_mean.npy")
        self.ae_scale = np.load(f"{models_dir}/ae_scaler_scale.npy")

        # Min/max bounds captured from the training data's actual score
        # distributions - used to normalize scores into a comparable 0-1
        # range at inference time, same idea as ensemble.py but without
        # needing the full training set at inference time.
        self.if_score_min, self.if_score_max = -0.31, 0.18
        self.ae_score_min, self.ae_score_max = 0.0, 25.0

        print("ONNX models loaded successfully!")

    def _scale(self, features: np.ndarray, mean, scale):
        return (features - mean) / scale

    def score(self, feature_dict: dict) -> dict:
        """Takes a dict of the 8 features and returns risk scores."""
        raw = np.array([[feature_dict[col] for col in FEATURE_COLS]], dtype=np.float32)

        # --- Isolation Forest ---
        if_input = self._scale(raw, self.if_mean, self.if_scale).astype(np.float32)
        if_output = self.if_session.run(None, {"input": if_input})
        # sklearn-onnx's IsolationForest outputs [label, score] - score is at index 1
        if_raw_score = float(if_output[1][0][0]) if len(if_output) > 1 else float(if_output[0][0])
        if_norm = np.clip(
            (-if_raw_score - self.if_score_min) / (self.if_score_max - self.if_score_min), 0, 1
        )

        # --- Autoencoder ---
        ae_input = self._scale(raw, self.ae_mean, self.ae_scale).astype(np.float32)
        ae_output = self.ae_session.run(None, {"input": ae_input})
        reconstructed = ae_output[0][0]
        reconstruction_error = float(np.mean((ae_input[0] - reconstructed) ** 2))
        ae_norm = np.clip(
            (reconstruction_error - self.ae_score_min) / (self.ae_score_max - self.ae_score_min), 0, 1
        )

        # --- Ensemble: simple average, same approach as ensemble.py ---
        risk_score = round(float((if_norm + ae_norm) / 2), 4)
        is_flagged = risk_score >= 0.5   # simple cutoff for now, tunable later

        return {
            "risk_score": risk_score,
            "is_flagged": is_flagged,
            "if_score": round(float(if_norm), 4),
            "ae_score": round(float(ae_norm), 4),
        }


async def compute_features(db, sender_account: str, amount: float) -> dict:
    """
    Rebuilds the same 8 features used in training, but for a SINGLE new
    transaction using only that sender's recent history from MongoDB -
    the real-time equivalent of the rolling-window logic in feature_engineering.py.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=1)

    # Pull this sender's transactions from the last hour
    cursor = db.transactions.find({
        "sender_account": sender_account,
        "timestamp": {"$gte": window_start}
    })
    recent_txns = await cursor.to_list(length=1000)

    txn_count_1h = len(recent_txns) + 1  # +1 to include the current transaction
    unique_receivers_1h = len(set(t["receiver_account"] for t in recent_txns)) + 1
    total_sent_1h = sum(t["amount"] for t in recent_txns) + amount

    # All-time average for this sender (expanding mean, same idea as training)
    all_time_cursor = db.transactions.find({"sender_account": sender_account})
    all_txns = await all_time_cursor.to_list(length=10000)
    all_amounts = [t["amount"] for t in all_txns] + [amount]
    avg_amount_sender = sum(all_amounts) / len(all_amounts)

    amount_deviation = (amount - avg_amount_sender) / avg_amount_sender if avg_amount_sender else 0

    near_threshold_flag = int(
        THRESHOLD * NEAR_THRESHOLD_RATIO <= amount < THRESHOLD
    )

    # Total received by this account (in_out_ratio) - across all time
    received_cursor = db.transactions.find({"receiver_account": sender_account})
    received_txns = await received_cursor.to_list(length=10000)
    total_received = sum(t["amount"] for t in received_txns)
    in_out_ratio = total_received / total_sent_1h if total_sent_1h else 0

    return {
        "amount": amount,
        "txn_count_1h": txn_count_1h,
        "unique_receivers_1h": unique_receivers_1h,
        "total_sent_1h": total_sent_1h,
        "avg_amount_sender": avg_amount_sender,
        "amount_deviation": amount_deviation,
        "near_threshold_flag": near_threshold_flag,
        "in_out_ratio": in_out_ratio,
    }
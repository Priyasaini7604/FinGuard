"""
FinGuard - Feature Engineering
Transforms raw transaction data into behavioral features that expose
structuring/smurfing patterns, using a rolling time window per sender account.
"""

import pandas as pd
import numpy as np

THRESHOLD = 50000            # reporting threshold banks watch for
NEAR_THRESHOLD_RATIO = 0.95  # "near threshold" = within 95% of the limit
WINDOW = "1h"                # rolling window size for behavioral features


def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_sender_features(df):
    """
    For every transaction, compute rolling-window stats about the SENDER's
    recent behavior (frequency, fan-out, amount patterns).

    We loop account-by-account so the rolling window only looks at that
    account's own transaction history - not the whole dataset.
    """
    df = df.set_index("timestamp")
    feature_frames = []

    for account, group in df.groupby("sender_account"):
        group = group.sort_index()

        # Rolling count of transactions in the past window (fan-out frequency)
        txn_count = group["amount"].rolling(WINDOW).count()

        # Rolling unique receivers - approximate via rolling apply
        # (how many DIFFERENT people this account paid recently)
        # pandas rolling() only works on numeric data, so we encode each
        # receiver_account string as a numeric code first, then count
        # unique codes within the window.
        receiver_codes = group["receiver_account"].astype("category").cat.codes.astype(float)
        unique_receivers = receiver_codes.rolling(WINDOW).apply(
            lambda x: pd.Series(x).nunique(), raw=False
        )

        # Rolling sum of money sent (total outflow in window)
        total_sent = group["amount"].rolling(WINDOW).sum()

        # Rolling average amount sent by this account historically
        avg_amount = group["amount"].expanding().mean()

        group = group.copy()
        group["sender_account"] = account
        group["txn_count_1h"] = txn_count.values
        group["unique_receivers_1h"] = unique_receivers.values
        group["total_sent_1h"] = total_sent.values
        group["avg_amount_sender"] = avg_amount.values

        feature_frames.append(group)

    result = pd.concat(feature_frames).sort_index()
    return result.reset_index()


def add_derived_features(df):
    """Add features that don't need rolling windows - just math on existing columns."""

    # How far is this transaction's amount from the sender's own average?
    # A big positive spike often signals unusual behavior.
    df["amount_deviation"] = (
        (df["amount"] - df["avg_amount_sender"]) / df["avg_amount_sender"].replace(0, np.nan)
    ).fillna(0)

    # Is this amount suspiciously close to the reporting threshold?
    # 1 = yes (classic structuring red flag), 0 = no
    df["near_threshold_flag"] = (
        df["amount"] >= THRESHOLD * NEAR_THRESHOLD_RATIO
    ).astype(int) & (df["amount"] < THRESHOLD).astype(int)

    # In/out ratio per account: how much they've received vs sent overall.
    # Mule accounts typically show money flowing OUT almost as fast as it comes IN.
    total_in = df.groupby("receiver_account")["amount"].sum()
    total_out = df.groupby("sender_account")["amount"].sum()

    df["total_received_by_sender"] = df["sender_account"].map(total_in).fillna(0)
    df["in_out_ratio"] = (
        df["total_received_by_sender"] / df["total_sent_1h"].replace(0, np.nan)
    ).fillna(0)

    return df


def main():
    print("Loading transaction data...")
    df = load_data()

    print("Computing rolling sender behavior features (this may take a bit)...")
    df = compute_sender_features(df)

    print("Adding derived features...")
    df = add_derived_features(df)

    # Fill any remaining NaNs (first transaction of an account has no history yet)
    feature_cols = [
        "txn_count_1h", "unique_receivers_1h", "total_sent_1h",
        "avg_amount_sender", "amount_deviation", "near_threshold_flag",
        "in_out_ratio"
    ]
    df[feature_cols] = df[feature_cols].fillna(0)

    output_path = "data/features.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone! Saved feature-engineered dataset to {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFeature summary for fraud vs normal transactions:")
    print(df.groupby("is_fraud")[feature_cols].mean())


if __name__ == "__main__":
    main()
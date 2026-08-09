"""
FinGuard - Synthetic Transaction Data Generator
Generates realistic banking transactions with injected structuring/smurfing
fraud patterns for training the AML detection model.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TOTAL_TRANSACTIONS = 50000
NUM_ACCOUNTS = 3000
FRAUD_RATIO = 0.03          # ~3% of transactions will be part of fraud rings
STRUCTURING_THRESHOLD = 50000   # amount banks usually flag above this (INR)
TRANSACTION_TYPES = ["NEFT", "IMPS", "UPI", "RTGS"]

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)


def random_timestamp(start, end):
    """Pick a random datetime between two datetimes."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_accounts(n):
    """Create a pool of fake account IDs."""
    return [f"ACC{str(i).zfill(5)}" for i in range(1, n + 1)]


def generate_normal_transactions(accounts, count):
    """
    Generate everyday, non-fraudulent transactions.
    Amounts follow a realistic skewed distribution -
    most transactions are small, few are large (like real spending habits).
    """
    rows = []
    for _ in range(count):
        sender, receiver = random.sample(accounts, 2)
        # Log-normal distribution gives realistic skew: many small txns, few large
        amount = round(np.random.lognormal(mean=8.5, sigma=1.0), 2)
        amount = min(amount, 500000)  # cap unrealistic outliers

        rows.append({
            "transaction_id": None,  # filled later
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "timestamp": random_timestamp(START_DATE, END_DATE),
            "transaction_type": random.choice(TRANSACTION_TYPES),
            "is_fraud": 0
        })
    return rows


def generate_structuring_ring(accounts):
    """
    Simulate one 'structuring' fraud ring:
    A mastermind account splits a large sum into many small transfers
    to mule accounts within a short time window, staying just under
    the reporting threshold to avoid detection.
    """
    rows = []
    mastermind = random.choice(accounts)
    num_splits = random.randint(5, 15)
    mule_accounts = random.sample(
        [a for a in accounts if a != mastermind], num_splits
    )

    # Fraud bursts happen fast - within a few hours, not spread over months
    ring_start = random_timestamp(START_DATE, END_DATE)

    for mule in mule_accounts:
        # Amount just under the threshold, with some randomness so it's not too obvious
        amount = round(random.uniform(
            STRUCTURING_THRESHOLD * 0.7, STRUCTURING_THRESHOLD * 0.98
        ), 2)
        txn_time = ring_start + timedelta(minutes=random.randint(1, 180))

        rows.append({
            "transaction_id": None,
            "sender_account": mastermind,
            "receiver_account": mule,
            "amount": amount,
            "timestamp": txn_time,
            "transaction_type": random.choice(["NEFT", "IMPS"]),  # fast rails preferred by fraudsters
            "is_fraud": 1
        })
    return rows


def main():
    print("Generating account pool...")
    accounts = generate_accounts(NUM_ACCOUNTS)

    fraud_txn_target = int(TOTAL_TRANSACTIONS * FRAUD_RATIO)
    normal_txn_target = TOTAL_TRANSACTIONS - fraud_txn_target

    print(f"Generating {normal_txn_target} normal transactions...")
    all_rows = generate_normal_transactions(accounts, normal_txn_target)

    print(f"Generating fraud rings to reach ~{fraud_txn_target} fraud transactions...")
    fraud_rows = []
    while len(fraud_rows) < fraud_txn_target:
        fraud_rows.extend(generate_structuring_ring(accounts))
    all_rows.extend(fraud_rows)

    print("Shuffling and assigning transaction IDs...")
    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)
    df["transaction_id"] = [f"TXN{str(i).zfill(6)}" for i in range(1, len(df) + 1)]

    # Reorder columns nicely
    df = df[[
        "transaction_id", "sender_account", "receiver_account",
        "amount", "timestamp", "transaction_type", "is_fraud"
    ]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone! Saved {len(df)} transactions to {output_path}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Unique accounts used: {pd.concat([df['sender_account'], df['receiver_account']]).nunique()}")


if __name__ == "__main__":
    main()
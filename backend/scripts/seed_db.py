"""
FinGuard - Database Seeding Script
Loads synthetic transactions + their pre-computed risk scores from CSV,
selects a representative subset, and inserts them into MongoDB Atlas
so the dashboard has realistic data to render (graph, flagged list, etc.)

Run this once from the backend folder:
    python scripts/seed_db.py
"""

import os
import certifi
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Same BASE_DIR pattern as main.py - resolved from this file's own location
# so it works regardless of which folder the script is run from.
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # scripts -> backend -> finguard
DATA_DIR = BASE_DIR / "ai-engine" / "data"

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "finguard")

# How many non-fraud transactions to sample (all fraud ones are kept regardless)
NORMAL_SAMPLE_SIZE = 800
RANDOM_SEED = 42  # fixed seed so re-running gives the same sample (reproducible demo)


def load_and_merge_data() -> pd.DataFrame:
    """Load transactions + their final risk scores, joined on transaction_id."""
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    risk_scores = pd.read_csv(DATA_DIR / "final_risk_scores.csv")

    # Only need transaction_id (join key), risk_score, predicted_fraud from the scores file -
    # the rest (is_fraud, if_anomaly_score, etc.) were intermediate model outputs, not needed in Mongo.
    risk_scores = risk_scores[["transaction_id", "risk_score", "predicted_fraud"]]

    merged = transactions.merge(risk_scores, on="transaction_id", how="inner")
    print(f"Loaded {len(transactions)} transactions, merged with risk scores -> {len(merged)} rows")
    return merged


def select_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all fraud-flagged transactions + a random sample of normal ones."""
    fraud_txns = df[df["predicted_fraud"] == 1]
    normal_txns = df[df["predicted_fraud"] == 0]

    sample_size = min(NORMAL_SAMPLE_SIZE, len(normal_txns))
    normal_sample = normal_txns.sample(n=sample_size, random_state=RANDOM_SEED)

    subset = pd.concat([fraud_txns, normal_sample]).reset_index(drop=True)
    print(f"Selected subset: {len(fraud_txns)} flagged + {len(normal_sample)} normal = {len(subset)} total")
    return subset


def to_mongo_documents(df: pd.DataFrame) -> list[dict]:
    """Convert each row into a document matching the TransactionOut schema shape."""
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "transaction_id": row["transaction_id"],
            "sender_account": row["sender_account"],
            "receiver_account": row["receiver_account"],
            "amount": float(row["amount"]),
            "transaction_type": row["transaction_type"],
            "timestamp": pd.to_datetime(row["timestamp"]).to_pydatetime(),
            "risk_score": float(row["risk_score"]),
            "is_flagged": bool(row["predicted_fraud"]),
        })
    return docs


def main():
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI not found - check your .env file")

    df = load_and_merge_data()
    subset = select_subset(df)
    documents = to_mongo_documents(subset)

    print("Connecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    db = client[DATABASE_NAME]
    collection = db["transactions"]

    before_count = collection.count_documents({})
    result = collection.insert_many(documents)
    after_count = collection.count_documents({})

    print(f"Inserted {len(result.inserted_ids)} documents.")
    print(f"Collection count: {before_count} -> {after_count}")

    client.close()


if __name__ == "__main__":
    main()
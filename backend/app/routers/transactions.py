"""
FinGuard - Transaction Router
API endpoints for submitting and retrieving transactions.
"""

from fastapi import APIRouter, Request
from datetime import datetime, timezone
import uuid

from app.models.schemas import TransactionIn, TransactionOut
from app.services.scoring import compute_features

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("/", response_model=TransactionOut)
async def create_transaction(transaction: TransactionIn, request: Request):
    """
    Receives a new transaction, computes real-time behavioral features from
    this sender's recent history, scores it with the ONNX models, saves the
    full result to MongoDB, and returns it.
    """
    db = request.app.mongodb
    scorer = request.app.fraud_scorer

    # Real-time equivalent of feature_engineering.py's rolling window logic -
    # looks at this sender's actual history in MongoDB right now.
    features = await compute_features(db, transaction.sender_account, transaction.amount)

    # Run both ONNX models and combine into a single risk score
    result = scorer.score(features)

    transaction_doc = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
        "sender_account": transaction.sender_account,
        "receiver_account": transaction.receiver_account,
        "amount": transaction.amount,
        "transaction_type": transaction.transaction_type,
        "timestamp": datetime.now(timezone.utc),
        "risk_score": result["risk_score"],
        "is_flagged": result["is_flagged"],
    }

    await db.transactions.insert_one(transaction_doc)

    return transaction_doc

@router.get("/stats/summary")
async def get_transaction_stats(request: Request):
    """
    High-level summary numbers for the dashboard's stat cards -
    total volume, how much is flagged, and how much money that represents.
    """
    db = request.app.mongodb

    total_count = await db.transactions.count_documents({})
    flagged_count = await db.transactions.count_documents({"is_flagged": True})

    # Sum the amount of flagged transactions using MongoDB's aggregation pipeline -
    # much faster than pulling every document into Python and summing there.
    pipeline = [
        {"$match": {"is_flagged": True}},
        {"$group": {"_id": None, "total_risk_amount": {"$sum": "$amount"}}},
    ]
    result = await db.transactions.aggregate(pipeline).to_list(length=1)
    total_risk_amount = result[0]["total_risk_amount"] if result else 0

    # Count unique accounts across both sender and receiver fields
    unique_senders = await db.transactions.distinct("sender_account")
    unique_receivers = await db.transactions.distinct("receiver_account")
    unique_accounts = len(set(unique_senders) | set(unique_receivers))

    return {
        "total_transactions": total_count,
        "flagged_transactions": flagged_count,
        "flagged_percentage": round((flagged_count / total_count) * 100, 1) if total_count else 0,
        "total_risk_amount": round(total_risk_amount, 2),
        "unique_accounts": unique_accounts,
    }

@router.get("/graph/data")
async def get_transaction_graph(request: Request, limit: int = 300):
    """
    Builds a node/edge graph from transactions - accounts become nodes,
    transactions become edges. Prioritizes flagged transactions since
    those are the most important ones to visualize; fills remaining
    slots with recent normal transactions to keep the graph readable.
    """
    db = request.app.mongodb

    # Highest-risk flagged transactions first
        # Split the budget: roughly 40% flagged (to clearly show suspicious
    # clusters) and 60% normal (so there's real contrast to compare against).
    flagged_limit = int(limit * 0.4)
    normal_limit = limit - flagged_limit

    flagged_cursor = db.transactions.find({"is_flagged": True}) \
        .sort("risk_score", -1).limit(flagged_limit)
    flagged_txns = await flagged_cursor.to_list(length=flagged_limit)

    normal_cursor = db.transactions.find({"is_flagged": False}) \
        .sort("timestamp", -1).limit(normal_limit)
    normal_txns = await normal_cursor.to_list(length=normal_limit)

    all_txns = flagged_txns + normal_txns

    # Build nodes: one entry per unique account, keeping the highest
    # risk_score seen across any transaction that account was part of.
    nodes = {}
    edges = []

    for txn in all_txns:
        sender = txn["sender_account"]
        receiver = txn["receiver_account"]
        risk = txn.get("risk_score", 0.0)
        flagged = txn.get("is_flagged", False)

        for account in (sender, receiver):
            if account not in nodes:
                nodes[account] = {"id": account, "risk_score": risk, "is_flagged": flagged}
            else:
                # Keep the account's worst-case risk score
                if risk > nodes[account]["risk_score"]:
                    nodes[account]["risk_score"] = risk
                    nodes[account]["is_flagged"] = flagged

        edges.append({
            "source": sender,
            "target": receiver,
            "amount": txn["amount"],
            "risk_score": risk,
            "is_flagged": flagged,
            "transaction_id": txn["transaction_id"],
        })

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


@router.get("/{transaction_id}", response_model=TransactionOut)
async def get_transaction(transaction_id: str, request: Request):
    """Fetch a single transaction by its ID - useful for checking a scored result."""
    db = request.app.mongodb
    doc = await db.transactions.find_one({"transaction_id": transaction_id})

    if doc is None:
        return {"error": "Transaction not found"}

    return doc



@router.get("/")
async def list_transactions(request: Request, limit: int = 20, flagged_only: bool = False):
    """List the most recent transactions - handy for a quick sanity check."""
    db = request.app.mongodb

    query = {"is_flagged": True} if flagged_only else {}
    cursor = db.transactions.find(query).sort("timestamp", -1).limit(limit)
    results = await cursor.to_list(length=limit)

    # Mongo's _id (ObjectId) isn't JSON-serializable by default, so drop it
    for doc in results:
        doc.pop("_id", None)

    return {"count": len(results), "transactions": results}
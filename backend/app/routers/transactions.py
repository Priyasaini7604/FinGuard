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
    flagged_cursor = db.transactions.find({"is_flagged": True}) \
        .sort("risk_score", -1).limit(limit)
    flagged_txns = await flagged_cursor.to_list(length=limit)

    # Fill any remaining space with recent normal transactions
    remaining = limit - len(flagged_txns)
    normal_txns = []
    if remaining > 0:
        normal_cursor = db.transactions.find({"is_flagged": False}) \
            .sort("timestamp", -1).limit(remaining)
        normal_txns = await normal_cursor.to_list(length=remaining)

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
async def list_transactions(request: Request, limit: int = 20):
    """List the most recent transactions - handy for a quick sanity check."""
    db = request.app.mongodb
    cursor = db.transactions.find().sort("timestamp", -1).limit(limit)
    results = await cursor.to_list(length=limit)

    # Mongo's _id (ObjectId) isn't JSON-serializable by default, so drop it
    for doc in results:
        doc.pop("_id", None)

    return {"count": len(results), "transactions": results}
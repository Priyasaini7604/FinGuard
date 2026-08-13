"""
FinGuard - Transaction Router
API endpoints for submitting and retrieving transactions.
"""

from fastapi import APIRouter, Request
from datetime import datetime, timezone
import uuid

from app.models.schemas import TransactionIn, TransactionOut

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("/", response_model=TransactionOut)
async def create_transaction(transaction: TransactionIn, request: Request):
    """
    Receives a new transaction, saves it to MongoDB, and returns it back.
    Risk scoring will be added here in the next step - for now this just
    proves the save/retrieve pipeline works end-to-end.
    """
    db = request.app.mongodb

    # Server generates these - never trust the client for identity/timing fields
    transaction_doc = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
        "sender_account": transaction.sender_account,
        "receiver_account": transaction.receiver_account,
        "amount": transaction.amount,
        "transaction_type": transaction.transaction_type,
        "timestamp": datetime.now(timezone.utc),
    }

    await db.transactions.insert_one(transaction_doc)

    return transaction_doc


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
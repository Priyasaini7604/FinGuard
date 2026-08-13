"""
FinGuard - Transaction Schemas
Pydantic models define the "shape" of data flowing in and out of the API.
FastAPI uses these to auto-validate requests and generate API docs.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class TransactionIn(BaseModel):
    """
    What the client sends us when reporting a new transaction.
    We don't ask for transaction_id or timestamp - the server generates
    those, so a client can't spoof them.
    """
    sender_account: str = Field(..., examples=["ACC00099"])
    receiver_account: str = Field(..., examples=["ACC00201"])
    amount: float = Field(..., gt=0, examples=[45000.0])
    transaction_type: str = Field(default="UPI", examples=["NEFT", "IMPS", "UPI", "RTGS"])


class TransactionOut(BaseModel):
    """
    What we send back after processing a transaction - includes the
    risk assessment alongside the original transaction details.
    """
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: str
    timestamp: datetime

    # Risk fields - filled in once we add scoring in the next step.
    # Optional for now so this schema works before scoring exists.
    risk_score: Optional[float] = None
    is_flagged: Optional[bool] = None
"""
FinGuard Backend - Main Application Entry Point
FastAPI app that connects to MongoDB and will serve fraud-detection endpoints.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
import os
import certifi
from pathlib import Path
from dotenv import load_dotenv

from app.routers import transactions
from app.services.scoring import FraudScorer

load_dotenv()

# Absolute path to ai-engine/models, calculated from this file's own location -
# this way it works no matter what directory uvicorn was launched from
# (uvicorn --reload can spawn subprocesses with a different working directory).
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # -> finguard/backend -> finguard
MODELS_DIR = BASE_DIR / "ai-engine" / "models"

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "finguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts, and once when it shuts down.
    We open the MongoDB connection here (not per-request) because creating
    a new connection for every API call would be slow and wasteful -
    motor's client manages a connection pool internally that we reuse.
    """
    print("Starting up: connecting to MongoDB...")
    # tlsCAFile explicitly points to certifi's trusted certificate bundle.
    # Windows sometimes uses an outdated/incompatible system CA store,
    # which causes the SSL handshake to fail against Atlas's servers.
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
    app.mongodb = app.mongodb_client[DATABASE_NAME]

    # Load ONNX models once here - loading them fresh on every request
    # would add noticeable latency and defeats the point of ONNX being fast.
    app.fraud_scorer = FraudScorer(models_dir=str(MODELS_DIR))

    # Quick check that the connection actually works
    try:
        await app.mongodb_client.admin.command("ping")
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

    yield  # server runs here, handling requests

    print("Shutting down: closing MongoDB connection...")
    app.mongodb_client.close()


app = FastAPI(title="FinGuard API", lifespan=lifespan)
app.include_router(transactions.router)


@app.get("/")
async def root():
    return {"message": "FinGuard API is running"}


@app.get("/health")
async def health_check():
    """Basic check that both the API and DB connection are alive."""
    try:
        await app.mongodb_client.admin.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {"api_status": "ok", "database_status": db_status}
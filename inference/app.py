import os
import json
import time
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import torch
import torch.nn as nn
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection Inference Service")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Model Definition
class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# Model Role selection
MODEL_ROLE = os.getenv("MODEL_ROLE", "baseline").lower()
MODELS_DIR = "/app/models"
DEVICE = torch.device("cpu") # Force CPU to avoid GPU resource locks

model = FraudNet().to(DEVICE)
model.eval()

def load_model():
    """
    Dynamically loads the appropriate model state dict from the shared volume.
    Candidate service falls back to baseline if candidate model isn't trained yet.
    """
    model_loaded = False
    
    # Identify model paths
    candidate_path = os.path.join(MODELS_DIR, "model_candidate.pt")
    baseline_path = os.path.join(MODELS_DIR, "model_baseline.pt")
    
    selected_path = None
    if MODEL_ROLE == "candidate":
        if os.path.exists(candidate_path):
            selected_path = candidate_path
        else:
            logger.info("Candidate model not found yet. Falling back to baseline model.")
            selected_path = baseline_path
    else:
        selected_path = baseline_path

    if selected_path and os.path.exists(selected_path):
        try:
            # Load state dict safely on CPU
            state_dict = torch.load(selected_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {selected_path} (role: {MODEL_ROLE})")
            model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model from {selected_path}: {e}")
    
    if not model_loaded:
        logger.warning(f"Using uninitialized/random weights for role: {MODEL_ROLE}")

# Initial load
load_model()

# Last reload check time to periodically look for new models (every 5 seconds)
last_reload_time = time.time()

class TransactionRequest(BaseModel):
    amount: float
    distance: float
    velocity: float
    age: float
    risk_score: float

class PredictionResponse(BaseModel):
    is_fraud: int
    probability: float
    model_role: str
    model_version: str # Used by simulator to verify dynamic hot-swaps

@app.get("/health")
def health():
    return {"status": "healthy", "role": MODEL_ROLE}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    global last_reload_time
    
    # Periodically reload candidate model if we are candidate role and a new one exists
    current_time = time.time()
    if MODEL_ROLE == "candidate" and current_time - last_reload_time > 5.0:
        load_model()
        last_reload_time = current_time

    features = [
        request.amount,
        request.distance,
        request.velocity,
        request.age,
        request.risk_score
    ]
    
    # Inference
    x = torch.tensor([features], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob = model(x).item()
        pred = 1 if prob >= 0.5 else 0

    # Log to Redis for statistical drift monitoring
    if redis_client:
        try:
            telemetry_data = {
                "timestamp": current_time,
                "features": features,
                "prediction": pred,
                "probability": prob
            }
            redis_client.rpush("telemetry_queue", json.dumps(telemetry_data))
            # Maintain a max size for safety in long runs
            redis_client.ltrim("telemetry_queue", -10000, -1)
        except Exception as e:
            logger.error(f"Error writing telemetry to Redis: {e}")

    # Determine version label to return
    # If candidate model path exists, version is candidate, else it is baseline
    model_version = MODEL_ROLE
    if MODEL_ROLE == "candidate":
        if os.path.exists(os.path.join(MODELS_DIR, "model_candidate.pt")):
            model_version = "candidate"
        else:
            model_version = "baseline"

    return PredictionResponse(
        is_fraud=pred,
        probability=prob,
        model_role=MODEL_ROLE,
        model_version=model_version
    )

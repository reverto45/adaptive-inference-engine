import os
import json
import time
import logging
from typing import Dict, List, Tuple
import numpy as np
import redis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

from worker.metrics import (
    start_metrics_exporter,
    increment_telemetry,
    set_psi,
    set_auc,
    set_ewc_penalty
)

# Configure structured JSON-like logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL", "5.0"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODELS_DIR = "/app/models"
ENVOY_CONF_DIR = os.getenv("ENVOY_CONF_DIR", "/etc/envoy")
DEVICE = torch.device("cpu") # Force CPU usage for retraining as it is lightweight and avoids GPU conflicts

# Model Definition matching Inference
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

# Statistical monitoring functions
def calculate_psi(reference: np.ndarray, production: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) for a feature.
    """
    # Use percentiles of reference to define bins
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference, percentiles)
    bins = np.unique(bins) # Remove duplicate bin edges
    
    if len(bins) < 2:
        return 0.0
    
    ref_counts, _ = np.histogram(reference, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)
    
    # Calculate proportions
    ref_props = ref_counts / len(reference)
    prod_props = prod_counts / len(production)
    
    # Adjust 0s to prevent division/log errors
    eps = 1e-4
    ref_props = np.where(ref_props == 0, eps, ref_props)
    prod_props = np.where(prod_props == 0, eps, prod_props)
    
    psi = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))
    return float(psi)

def calculate_adversarial_auc(reference: np.ndarray, production: np.ndarray) -> float:
    """
    Calculate the Adversarial Validation AUC to measure similarity between ref and prod datasets.
    """
    ref_labels = np.zeros(len(reference))
    prod_labels = np.ones(len(production))
    
    X = np.vstack([reference, production])
    y = np.concatenate([ref_labels, prod_labels])
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return float(auc)
    except Exception as e:
        logger.error(f"Error calculating adversarial AUC: {e}")
        return 0.5

# Elastic Weight Consolidation (EWC) implementation
def compute_fisher(model: nn.Module, X_ref: torch.Tensor, y_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal elements of the Fisher Information Matrix on the reference dataset.
    """
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)
        
    model.eval()
    criterion = nn.BCELoss()
    
    for i in range(len(X_ref)):
        x_val = X_ref[i].unsqueeze(0)
        y_val = y_ref[i].unsqueeze(0)
        
        model.zero_grad()
        output = model(x_val)
        loss = criterion(output, y_val)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.data ** 2) / len(X_ref)
                
    return fisher

def retrain_model_ewc(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_drift: np.ndarray,
    y_drift: np.ndarray,
    ewc_lambda: float = 800.0,
    epochs: int = 20
) -> FraudNet:
    """
    Retrain the model on the drifted dataset using Elastic Weight Consolidation (EWC).
    """
    logger.info("Initializing EWC Smart Retraining Loop...")
    
    # Initialize baseline model and load current weights
    baseline_model = FraudNet().to(DEVICE)
    baseline_path = os.path.join(MODELS_DIR, "model_baseline.pt")
    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        logger.info("Loaded baseline model weights for EWC.")
    else:
        logger.warning("Baseline model not found. Initializing random weights.")
        
    # Convert numpy arrays to tensors
    X_ref_t = torch.tensor(X_ref, dtype=torch.float32).to(DEVICE)
    y_ref_t = torch.tensor(y_ref, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_drift_t = torch.tensor(X_drift, dtype=torch.float32).to(DEVICE)
    y_drift_t = torch.tensor(y_drift, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # Compute Fisher Information Matrix on reference data
    fisher = compute_fisher(baseline_model, X_ref_t, y_ref_t)
    
    # Prepare the model to be retrained
    candidate_model = FraudNet().to(DEVICE)
    candidate_model.load_state_dict(baseline_model.state_dict())
    candidate_model.train()
    
    optimizer = optim.Adam(candidate_model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    # Keep reference weights for EWC regularization
    baseline_params = {name: param.clone() for name, param in baseline_model.named_parameters()}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Base loss on new drifted data
        outputs = candidate_model(X_drift_t)
        base_loss = criterion(outputs, y_drift_t)
        
        # Calculate EWC loss penalty
        ewc_loss = 0.0
        for name, param in candidate_model.named_parameters():
            ewc_loss += (fisher[name] * (param - baseline_params[name]) ** 2).sum()
            
        total_loss = base_loss + (ewc_lambda / 2.0) * ewc_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Record EWC metrics
        set_ewc_penalty(active_epoch=f"epoch_{epoch}", value=float(ewc_loss.item()))
        logger.info(f"Epoch {epoch}/{epochs} | Base Loss: {base_loss.item():.4f} | EWC Loss: {ewc_loss.item():.4f} | Total: {total_loss.item():.4f}")
        
    return candidate_model

def hot_swap_envoy():
    """
    Rewrites cds.yaml atomically to reroute production traffic from baseline to candidate.
    """
    logger.info("Triggering Envoy dynamic hot-swap...")
    
    # Write a new CDS yaml pointing the baseline-service cluster endpoints to the candidate service container.
    # Note: Envoy will dynamically re-evaluate the upstream host.
    new_cds = """resources:
- "@type": type.googleapis.com/envoy.config.cluster.v3.Cluster
  name: mlops-engine-baseline-service
  connect_timeout: 0.25s
  type: STRICT_DNS
  lb_policy: ROUND_ROBIN
  load_assignment:
    cluster_name: mlops-engine-baseline-service
    endpoints:
      - lb_endpoints:
          - endpoint:
              address:
                socket_address:
                  address: candidate  # <-- Pointed from 'baseline' to 'candidate' for hot-swap
                  port_value: 8000
- "@type": type.googleapis.com/envoy.config.cluster.v3.Cluster
  name: mlops-engine-shadow-service
  connect_timeout: 0.25s
  type: STRICT_DNS
  lb_policy: ROUND_ROBIN
  load_assignment:
    cluster_name: mlops-engine-shadow-service
    endpoints:
      - lb_endpoints:
          - endpoint:
              address:
                socket_address:
                  address: candidate
                  port_value: 8000
"""
    cds_path = os.path.join(ENVOY_CONF_DIR, "cds.yaml")
    tmp_path = cds_path + ".tmp"
    
    try:
        # Atomic file write to avoid Envoy reading half-written config
        with open(tmp_path, "w") as f:
            f.write(new_cds)
        os.replace(tmp_path, cds_path)
        logger.info("Envoy cds.yaml written atomically and hot-swapped successfully.")
    except Exception as e:
        logger.error(f"Error during Envoy hot-swap: {e}")

def main():
    # Start metrics exporter on port 9090
    start_metrics_exporter(9090)
    
    # Wait for reference data to be written by the simulator
    ref_path = os.path.join(MODELS_DIR, "reference_data.json")
    logger.info(f"Waiting for reference data at {ref_path}...")
    while not os.path.exists(ref_path):
        time.sleep(2)
        
    try:
        with open(ref_path, "r") as f:
            ref_data = json.load(f)
        X_ref = np.array(ref_data["features"])
        y_ref = np.array(ref_data["labels"])
        logger.info(f"Loaded reference dataset: {X_ref.shape}")
    except Exception as e:
        logger.critical(f"Failed to load reference dataset: {e}")
        return

    # Connect to Redis
    logger.info(f"Connecting to Redis at {REDIS_URL}...")
    r = None
    for attempt in range(10):
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            logger.info("Connected to Redis.")
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    if not r:
        logger.critical("Could not connect to Redis. Exiting.")
        return

    logger.info("Entering monitoring loop...")
    drift_detected = False
    
    while True:
        try:
            # Check length of the queue
            queue_len = r.llen("telemetry_queue")
            logger.info(f"Current telemetry records in queue: {queue_len}")
            
            # If we have enough samples to perform drift detection (e.g. 50 samples)
            if queue_len >= 50 and not drift_detected:
                logger.info("Fetching telemetry data for drift assessment...")
                raw_records = r.lrange("telemetry_queue", 0, -1)
                
                records = [json.loads(rec) for rec in raw_records]
                X_prod = np.array([rec["features"] for rec in records])
                y_prod = np.array([rec["prediction"] for rec in records])
                
                # Increment records metric
                increment_telemetry(stream_source="production", status="processed", amount=len(records))
                
                # 1. Compute PSI for each feature
                # Feature indices: 0: amount, 1: distance, 2: velocity, 3: age, 4: risk_score
                feature_names = ["amount", "distance", "velocity", "age", "risk_score"]
                psi_values = {}
                max_psi = 0.0
                
                for idx, name in enumerate(feature_names):
                    psi = calculate_psi(X_ref[:, idx], X_prod[:, idx])
                    psi_values[name] = psi
                    set_psi(feature_name=name, value=psi)
                    max_psi = max(max_psi, psi)
                    logger.info(f"PSI for {name}: {psi:.4f}")
                    
                # 2. Compute Adversarial AUC
                auc = calculate_adversarial_auc(X_ref, X_prod)
                set_auc(snapshot_id=f"snapshot_{int(time.time())}", value=auc)
                logger.info(f"Adversarial Validation AUC-ROC: {auc:.4f}")
                
                # Check for drift triggers
                # Standard threshold: PSI > 0.2 or AUC > 0.7
                if max_psi > 0.25 or auc > 0.72:
                    logger.warning("DATA DRIFT DETECTED! Triggering adaptive mitigation pipeline.")
                    drift_detected = True
                    
                    # Retrain candidate model using EWC
                    candidate_model = retrain_model_ewc(X_ref, y_ref, X_prod, y_prod)
                    
                    # Save candidate model
                    candidate_path = os.path.join(MODELS_DIR, "model_candidate.pt")
                    torch.save(candidate_model.state_dict(), candidate_path)
                    logger.info(f"Candidate model saved to {candidate_path}")
                    
                    # Hot-swap Envoy config to route to candidate container
                    hot_swap_envoy()
                    
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

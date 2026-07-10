import logging
from collections import deque
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp
from prometheus_client import Gauge, start_http_server

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MLOps Analytics Worker")

# sliding window of last 1000 requests
WINDOW_SIZE = 1000
baseline_window = deque(maxlen=WINDOW_SIZE)
shadow_window = deque(maxlen=WINDOW_SIZE)

# Expose calculated metrics via Prometheus
divergence_score_gauge = Gauge(
    "shadow_model_divergence_score",
    "Kullback-Leibler (KL) Divergence or Kolmogorov-Smirnov (KS) test distance between baseline and shadow distributions"
)

class LogPayload(BaseModel):
    request_id: str = Field(..., description="Unique request ID")
    model_type: str = Field(..., description="Type of the model: 'baseline' or 'shadow'")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability (float between 0 and 1)")

def calculate_drift():
    """
    Computes a two-sample Kolmogorov-Smirnov (KS) test using scipy to compare baseline and shadow distributions.
    Sets the divergence score to the KS statistic.
    If either window has insufficient samples, we don't compute yet.
    """
    if len(baseline_window) < 100 or len(shadow_window) < 100:
        logger.info(f"Insufficient samples to compute KS test (baseline: {len(baseline_window)}, shadow: {len(shadow_window)})")
        return

    # Convert deques to lists/arrays for scipy
    b_data = list(baseline_window)
    s_data = list(shadow_window)

    try:
        # Run 2-sample KS test
        # ks_2samp returns (statistic, pvalue)
        result = ks_2samp(b_data, s_data)
        ks_stat = float(result.statistic)

        # Expose the metric
        divergence_score_gauge.set(ks_stat)
        logger.info(f"KS test computed. Divergence Score (KS statistic): {ks_stat:.4f}, p-value: {result.pvalue:.4f}")
    except Exception as e:
        logger.error(f"Error calculating KS test: {e}", exc_info=True)

@app.post("/logs")
def ingest_log(payload: LogPayload, background_tasks: BackgroundTasks):
    """
    Ingests streaming log payloads containing request_id, model_type, and predicted_probability.
    """
    prob = payload.predicted_probability
    model_type = payload.model_type.lower().strip()

    if model_type == "baseline":
        baseline_window.append(prob)
    elif model_type == "shadow" or model_type == "candidate":
        # Accept 'shadow' or 'candidate' as the shadow distribution
        shadow_window.append(prob)
    else:
        logger.warning(f"Unknown model_type received: {payload.model_type}")
        return {"status": "ignored", "reason": "unknown model_type"}

    # Compute drift in background to avoid blocking the ingestion response
    background_tasks.add_task(calculate_drift)

    return {"status": "accepted"}

@app.get("/metrics-summary")
def metrics_summary():
    """
    Helper endpoint to query current status of sliding windows.
    """
    b_len = len(baseline_window)
    s_len = len(shadow_window)
    return {
        "baseline_samples": b_len,
        "shadow_samples": s_len,
        "window_size": WINDOW_SIZE
    }

def start_metrics_server():
    # Start prometheus metrics exporter on port 9090
    try:
        start_http_server(9092) # Use 9092 to avoid conflicts with other worker ports
        logger.info("Prometheus metrics server started on port 9092")
    except Exception as e:
        logger.error(f"Failed to start prometheus server on 9092: {e}")

# Start metrics server on startup
@app.on_event("startup")
def startup_event():
    start_metrics_server()

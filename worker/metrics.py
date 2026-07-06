"""
worker/metrics.py

Prometheus instrumentation helper for the MLOps pipeline.

Exports:
- mlops_telemetry_records_total (Counter) labels: ['stream_source', 'status']
- mlops_population_stability_index (Gauge) labels: ['feature_name']
- mlops_adversarial_validation_auc (Gauge) labels: ['snapshot_id']
- mlops_ewc_loss_penalty_magnitude (Gauge) labels: ['active_epoch']

- start_metrics_exporter(port: int = 9090): starts prometheus HTTP exporter on background thread.

Usage example:
    from worker.metrics import start_metrics_exporter, increment_telemetry, set_psi, set_auc, set_ewc_penalty
    start_metrics_exporter(9090)
    increment_telemetry(stream_source="ingest", status="accepted")
    set_psi(feature_name="feature_0", value=0.12)
    set_auc(snapshot_id="snapshot-2026-07-01", value=0.71)
    set_ewc_penalty(active_epoch="epoch_1", value=3.1415)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from prometheus_client import Counter, Gauge, start_http_server

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings.
    """
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Configure structured logging
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Counter for telemetry records processed/dropped
mlops_telemetry_records_total = Counter(
    "mlops_telemetry_records_total",
    "Total telemetry records handled by the pipeline",
    ["stream_source", "status"],
)

# Gauge for per-feature PSI
mlops_population_stability_index = Gauge(
    "mlops_population_stability_index",
    "Population Stability Index (PSI) per feature",
    ["feature_name"],
)

# Gauge for adversarial validation AUC-ROC
mlops_adversarial_validation_auc = Gauge(
    "mlops_adversarial_validation_auc",
    "Adversarial validation AUC-ROC per reservoir snapshot",
    ["snapshot_id"],
)

# Gauge for EWC penalty magnitude (inspect aggregated penalty or per-epoch magnitude)
mlops_ewc_loss_penalty_magnitude = Gauge(
    "mlops_ewc_loss_penalty_magnitude",
    "Elastic Weight Consolidation penalty magnitude per epoch",
    ["active_epoch"],
)


def start_metrics_exporter(port: int = 9090) -> None:
    """
    Start the prometheus_client HTTP server in a background thread so Prometheus can scrape metrics.

    Runs start_http_server which creates a background thread; this function is idempotent.
    """
    try:
        start_http_server(port)
        logger.info(f"Metrics exporter started on port {port}")
    except OSError as e:
        logger.error(f"Failed to start metrics exporter on port {port}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error starting metrics exporter: {e}")


# Helper functions to update metrics

def increment_telemetry(stream_source: str = "unknown", status: str = "accepted", amount: int = 1) -> None:
    """
    Increment telemetry records counter.

    status examples: accepted, dropped, failed, queued
    stream_source examples: ingress, reservoir, replay
    """
    try:
        mlops_telemetry_records_total.labels(stream_source=stream_source, status=status).inc(amount)
    except Exception as e:
        logger.error(f"Error incrementing telemetry metrics: {e}")


def set_psi(feature_name: str, value: float) -> None:
    """
    Set the PSI gauge for a feature.
    """
    try:
        mlops_population_stability_index.labels(feature_name=feature_name).set(float(value))
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for PSI metric: {e}")
    except Exception as e:
        logger.error(f"Error setting PSI metric: {e}")


def set_auc(snapshot_id: str, value: float) -> None:
    """
    Set adversarial validation AUC for a snapshot.
    """
    try:
        mlops_adversarial_validation_auc.labels(snapshot_id=snapshot_id).set(float(value))
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for AUC metric: {e}")
    except Exception as e:
        logger.error(f"Error setting AUC metric: {e}")


def set_ewc_penalty(active_epoch: str, value: float) -> None:
    """
    Record aggregated EWC penalty magnitude for an epoch (or step).
    """
    try:
        mlops_ewc_loss_penalty_magnitude.labels(active_epoch=active_epoch).set(float(value))
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for EWC penalty metric: {e}")
    except Exception as e:
        logger.error(f"Error setting EWC penalty metric: {e}")

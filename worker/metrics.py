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
    increment_telemetry("ingest", "accepted")
    set_psi("feature_0", 0.12)
    set_auc("snapshot-2026-07-01", 0.71)
    set_ewc_penalty("epoch_1", 3.1415)
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from prometheus_client import Counter, Gauge, start_http_server

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
    # start_http_server already spins in background; call it from here for convenience
    try:
        start_http_server(port)
    except Exception:
        # best-effort: the start_http_server may have already been called or port in use
        pass


# Helper functions to update metrics

def increment_telemetry(stream_source: str = "unknown", status: str = "accepted", amount: int = 1) -> None:
    """
    Increment telemetry records counter.

    status examples: accepted, dropped, failed, queued
    stream_source examples: ingress, reservoir, replay
    """
    mlops_telemetry_records_total.labels(stream_source, status).inc(amount)


def set_psi(feature_name: str, value: float) -> None:
    """
    Set the PSI gauge for a feature.
    """
    try:
        mlops_population_stability_index.labels(feature_name).set(float(value))
    except Exception:
        # ignore metric failures
        pass


def set_auc(snapshot_id: str, value: float) -> None:
    """
    Set adversarial validation AUC for a snapshot.
    """
    try:
        mlops_adversarial_validation_auc.labels(snapshot_id).set(float(value))
    except Exception:
        pass


def set_ewc_penalty(active_epoch: str, value: float) -> None:
    """
    Record aggregated EWC penalty magnitude for an epoch (or step).
    """
    try:
        mlops_ewc_loss_penalty_magnitude.labels(active_epoch).set(float(value))
    except Exception:
        pass

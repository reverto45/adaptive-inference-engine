# MLOps Engine Quickstart Demo

This demo allows you to run a simplified version of the MLOps Engine locally using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed.

## Getting Started

1. **Start the services**:
   ```bash
   docker-compose up -d
   ```

2. **Wait for initialization**:
   The services will initialize with a mock dataset and a pre-trained model.

3. **Observe the pipeline**:
   - The worker service will monitor for data drift using the provided `mock_data.csv`.
   - You can view Prometheus metrics at `http://localhost:9090`.
   - The Envoy gateway is accessible at `http://localhost:8080`.

4. **Trigger a Retraining (Simulation)**:
   You can simulate data drift by updating the `mock_data.csv` with different values and restarting the worker.

## Files

- `docker-compose.yaml`: Orchestrates the local demo.
- `mock_data.csv`: Sample dataset for drift monitoring.
- `mock_model.bin`: A placeholder for a pre-trained model.

# Grafana & LogQL Configuration Mockup and Kubernetes Snippets

This document provides visual configuration mockups, LogQL snippets for Grafana, and Kubernetes Deployment snippets matching the requirements of the enterprise MLOps platform.

---

## 1. Grafana and LogQL Overlaying Distribution Mockup

To compare the predicted probability distributions of the `baseline` and `shadow` models, we can leverage Grafana's **Histogram** or **Bar Gauge** panel with LogQL.

### LogQL Snippets for Distribution
Assuming your application logs structured JSON records to Loki with the model type and predicted probability:

#### Baseline Probability Distribution:
```logql
{container="mlops-baseline"} | json | line_format "{{.message}}" | __error__="" | unwrap predicted_probability
```

#### Shadow/Candidate Probability Distribution:
```logql
{container="mlops-candidate"} | json | line_format "{{.message}}" | __error__="" | unwrap predicted_probability
```

Alternatively, if parsing the HTTP payloads logged from Envoy or the applications:
```logql
{app="mlops-engine"} |= "predicted_probability" | json | model_type="baseline" | unwrap predicted_probability
```
```logql
{app="mlops-engine"} |= "predicted_probability" | json | model_type="shadow" | unwrap predicted_probability
```

### Grafana Alert Trigger Configuration
To trigger an alert if the divergence score calculated from the streaming distributions crosses the `0.05` threshold:

- **Metric**: `shadow_model_divergence_score`
- **Query / Formula**:
  ```promql
  shadow_model_divergence_score >= 0.05
  ```
- **Alert Rule Details**:
  - **Condition**: `A` (where query `A` is `shadow_model_divergence_score`)
  - **Evaluated**: Every `10s` for `1m`
  - **Threshold**: `0.05`
  - **Alert Name**: `Shadow Model Divergence Alert`
  - **Alert Description**: `Critical Alert: The Kolmogorov-Smirnov statistical divergence score between the baseline and shadow model distributions has exceeded 0.05 (current: {{ $values.A.Value }}). Hold hot-swap deployment.`

---

## 2. Robust QoS Kubernetes Deployment Snippet

Coupling the robust consumer with strict CPU and memory resource limits ensures OS-level Out-of-Memory (OOM) isolation without impacting core cluster nodes.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-retraining-worker
  namespace: mlops
  labels:
    app: retraining-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: retraining-worker
  template:
    metadata:
      labels:
        app: retraining-worker
    spec:
      containers:
        - name: ewc-consumer
          image: your-registry/mlops-worker:1.0.0
          imagePullPolicy: IfNotPresent
          env:
            - name: RMQ_URL
              valueFrom:
                secretKeyRef:
                  name: mlops-secrets
                  key: rabbitmq-url
            - name: RMQ_QUEUE
              value: "ewc_retraining_jobs"
          # Strict QoS guarantees OS-level isolation
          resources:
            requests:
              cpu: "1"
              memory: "1Gi"
            limits:
              cpu: "2"       # Cap at 2 CPUs
              memory: "2Gi"     # Cap at 2Gi memory to prevent OOM cascade
          securityContext:
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 10001
```

---

## 3. Envoy Model Readiness Probe Integration

Below is a snippet demonstrating how to configure the `readinessProbe` pointing to `/healthz/ready`, ensuring Envoy does not route traffic to a warming pod prematurely.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-baseline
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: baseline
          image: your-registry/mlops-engine:1.0.0
          ports:
            - containerPort: 8000
              name: http
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: http
            initialDelaySeconds: 5  # Allow application to boot and start warming background task
            periodSeconds: 2        # Frequently poll to detect when warming finishes
            timeoutSeconds: 2
            failureThreshold: 3
```

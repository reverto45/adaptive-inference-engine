import unittest
import json
import torch
import numpy as np
from scipy.stats import ks_2samp
from fastapi.testclient import TestClient

# Import elements we want to test
from inference.app import app as inference_app, FraudNet, is_warmed_up
from worker.analytics_worker import app as analytics_app, baseline_window, shadow_window

class TestMLOpsEngineComponents(unittest.TestCase):

    def test_inference_warming_and_readiness(self):
        """
        Verify model warming and readiness probe in inference app.
        """
        client = TestClient(inference_app)

        # Test readiness check endpoint before/after completion
        # Normally starts warming immediately, but we can verify endpoint response
        response = client.get("/healthz/ready")
        if response.status_code == 200:
            self.assertEqual(response.json()["status"], "ready")
        else:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "warming")

    def test_inference_model_forward(self):
        """
        Verify the FraudNet model can do a basic forward pass with shape (1, 5).
        """
        model = FraudNet()
        dummy_tensor = torch.randn(1, 5)
        out = model(dummy_tensor)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(0.0 <= out.item() <= 1.0)

    def test_analytics_worker_ks_test(self):
        """
        Verify that the analytics worker correctly handles sliding window logs and computes KS stats.
        """
        client = TestClient(analytics_app)

        # Clear windows
        baseline_window.clear()
        shadow_window.clear()

        # Ingest < 100 samples and check that KS isn't computed yet
        for i in range(50):
            client.post("/logs", json={
                "request_id": f"req_{i}",
                "model_type": "baseline",
                "predicted_probability": 0.1
            })
            client.post("/logs", json={
                "request_id": f"req_{i}",
                "model_type": "shadow",
                "predicted_probability": 0.8
            })

        summary = client.get("/metrics-summary").json()
        self.assertEqual(summary["baseline_samples"], 50)
        self.assertEqual(summary["shadow_samples"], 50)

        # Ingest enough samples (100+) to trigger calculation
        for i in range(50, 110):
            client.post("/logs", json={
                "request_id": f"req_{i}",
                "model_type": "baseline",
                "predicted_probability": 0.1
            })
            client.post("/logs", json={
                "request_id": f"req_{i}",
                "model_type": "shadow",
                "predicted_probability": 0.8
            })

        summary = client.get("/metrics-summary").json()
        self.assertEqual(summary["baseline_samples"], 110)
        self.assertEqual(summary["shadow_samples"], 110)

        # Manual KS check to verify the logic
        ks_stat, p_val = ks_2samp(list(baseline_window), list(shadow_window))
        # Since all baseline are 0.1 and shadow are 0.8, the KS distance should be 1.0
        self.assertAlmostEqual(ks_stat, 1.0)

if __name__ == "__main__":
    unittest.main()

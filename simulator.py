import os
import time
import json
import numpy as np
import requests
import torch
import torch.nn as nn

HEADERS = {"User-Agent": "AdaptiveInferenceEngineDemo/1.0 (foxprint666@users.noreply.github.com; python-requests)"}

# Model Definition matching containers
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

def fetch_live_wikimedia_events(num_events: int, filter_enwiki: bool = False, inject_drift: bool = False) -> list:
    """
    Connects to the live public Wikimedia EventStreams recent changes endpoint and extracts edit features.
    """
    events = []
    url = "https://stream.wikimedia.org/v2/stream/recentchange"
    
    print(f"Connecting to live Wikimedia EventStreams (fetching {num_events} events, enwiki_only={filter_enwiki})...")
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to connect to Wikimedia stream, status: {response.status_code}")
            
        for line in response.iter_lines():
            if len(events) >= num_events:
                break
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    try:
                        change = json.loads(decoded_line[6:])
                        
                        # 1. Filter by wiki if requested (e.g. enwiki for simulation, False for fast reference data harvest)
                        if filter_enwiki and change.get("wiki") != "enwiki":
                            continue
                            
                        # 2. Safe length_diff calculation
                        length_data = change.get("length", {})
                        new_len = length_data.get("new", 0)
                        old_len = length_data.get("old", 0)
                        length_diff = float(abs(new_len - old_len))
                        
                        # 3. Direct extraction of features
                        is_bot = 1.0 if change.get("bot", False) else 0.0
                        is_minor = 1.0 if change.get("minor", False) else 0.0
                        title_length = float(len(change.get("title", "")))
                        anon = 1.0 if change.get("anon", False) else 0.0
                        
                        features = [length_diff, is_bot, is_minor, title_length, anon]
                        
                        # 4. Inject drift features if requested (vandalism attack signature)
                        if inject_drift:
                            features[0] = features[0] + 1500.0  # Massive length difference
                            features[1] = 0.0                  # No bot edits
                            features[2] = 0.0                  # Not minor
                            features[3] = features[3] + 25.0    # Long title edits
                            features[4] = 1.0                  # 100% anonymous IP edits
                            
                        # Generate binary label (anomaly/vandalism classification target)
                        # Higher length changes, unregistered IP edits, non-minor increase probability of vandalism
                        logits = (features[0] * 0.005) + (features[4] * 2.0) - (features[1] * 3.0) - (features[2] * 1.5) + (features[3] * 0.02) - 1.0
                        prob = 1.0 / (1.0 + np.exp(-logits))
                        label = int(np.random.binomial(1, prob))
                        
                        events.append((features, label))
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error reading from live stream: {e}")
        
    return events

def generate_reference_data_and_baseline():
    print("Pre-generating reference dataset from Wikimedia live edits...")
    
    # Harvest 1000 events globally from all wikis (without filtering) for speed
    events = fetch_live_wikimedia_events(1000, filter_enwiki=False, inject_drift=False)
    
    if len(events) < 1000:
        print("Warning: Could not fetch enough live events. Generating synthetic fallbacks.")
        # Fallback to synthetic if stream was blocked
        np.random.seed(42)
        for _ in range(1000 - len(events)):
            features = [
                float(np.random.exponential(50.0)),
                1.0 if np.random.rand() < 0.15 else 0.0,
                1.0 if np.random.rand() < 0.3 else 0.0,
                float(np.random.normal(20, 8)),
                1.0 if np.random.rand() < 0.2 else 0.0
            ]
            label = 1 if (features[0] * 0.01 + features[4] * 1.5 - features[1] * 2.0) > 0.5 else 0
            events.append((features, label))
            
    features_list = [ev[0] for ev in events]
    labels_list = [ev[1] for ev in events]
    
    # Save reference data
    os.makedirs("shared-models", exist_ok=True)
    ref_path = os.path.join("shared-models", "reference_data.json")
    with open(ref_path, "w") as f:
        json.dump({
            "features": features_list,
            "labels": labels_list
        }, f)
    print(f"Reference data saved to {ref_path}")
    
    # Train baseline model
    X_train = torch.tensor(features_list, dtype=torch.float32)
    y_train = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
    
    model = FraudNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training baseline model (FraudNet) on live reference data...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")
            
    # Save baseline model weights
    baseline_path = os.path.join("shared-models", "model_baseline.pt")
    torch.save(model.state_dict(), baseline_path)
    print(f"Baseline model saved to {baseline_path}")

def run_simulation():
    url = "http://localhost:8080/predict"
    
    print("\nConnecting to Envoy Gateway at http://localhost:8080...")
    connected = False
    for attempt in range(30):
        try:
            r = requests.post(url, json={
                "amount": 10.0, "distance": 0.0, "velocity": 0.0, "age": 15.0, "risk_score": 0.0
            }, timeout=2)
            if r.status_code == 200:
                print("Envoy Gateway is online!")
                connected = True
                break
        except Exception:
            pass
        print(f"Waiting for Envoy (attempt {attempt+1}/30)...")
        time.sleep(2)
        
    if not connected:
        print("Could not connect to Envoy Gateway. Please verify docker-compose is running.")
        return

    # PHASE 1: Send Normal English Wikipedia Edits
    print("\n--- PHASE 1: Streaming Live English Wikipedia Recent Changes ---")
    
    # Ingest 60 live edits from enwiki
    events_p1 = fetch_live_wikimedia_events(60, filter_enwiki=True, inject_drift=False)
    
    for i, (features, label) in enumerate(events_p1):
        payload = {
            "amount": features[0],      # length_diff
            "distance": features[1],    # is_bot
            "velocity": features[2],    # is_minor
            "age": features[3],         # title_length
            "risk_score": features[4]   # anon
        }
        
        try:
            r = requests.post(url, json=payload, timeout=2)
            res = r.json()
            print(f"Edit {i+1}/60 | Live data: length_diff={features[0]:.1f}, bot={features[1]}, minor={features[2]}, title_len={features[3]:.0f}, anon={features[4]} | Version: {res['model_version']}")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
        time.sleep(0.15)  # Tiny throttle for readability

    # PHASE 2: Inject Coordinated Vandalism Attack (Drift)
    print("\n--- PHASE 2: Injecting Vandalism/Data Drift Attack ---")
    print("Simulating high-volume IP address edits with massive deletions/additions...")
    
    # Keep fetching live edits but mutate features to simulate drift
    url_stream = "https://stream.wikimedia.org/v2/stream/recentchange"
    swapped = False
    drift_req_count = 0
    
    try:
        response = requests.get(url_stream, headers=HEADERS, stream=True, timeout=15)
        for line in response.iter_lines():
            if drift_req_count >= 150 or swapped:
                break
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    try:
                        change = json.loads(decoded_line[6:])
                        if change.get("wiki") != "enwiki":
                            continue
                            
                        # Extract and immediately mutate/drift features
                        length_data = change.get("length", {})
                        new_len = length_data.get("new", 0)
                        old_len = length_data.get("old", 0)
                        length_diff = float(abs(new_len - old_len)) + 1500.0  # Drift shift!
                        
                        is_bot = 0.0  # Coordinated drift vandalism is never bot edits
                        is_minor = 0.0
                        title_length = float(len(change.get("title", ""))) + 25.0
                        anon = 1.0  # Coordinated drift vandalism from IP address users
                        
                        payload = {
                            "amount": length_diff,
                            "distance": is_bot,
                            "velocity": is_minor,
                            "age": title_length,
                            "risk_score": anon
                        }
                        
                        drift_req_count += 1
                        r = requests.post(url, json=payload, timeout=2)
                        res = r.json()
                        active_version = res['model_version']
                        
                        print(f"Edit {drift_req_count} (Drifted) | length_diff={length_diff:.1f}, bot={is_bot}, minor={is_minor}, title_len={title_length:.0f}, anon={anon} | Version: {active_version}")
                        
                        if active_version == "candidate":
                            print("\n[SUCCESS] DATA DRIFT DETECTED, MODEL RETRAINED (EWC), AND ENVOY GATEWAY ATOMICALLY HOT-SWAPPED TRAFFIC TO CANDIDATE CONTAINER WITH ZERO DOWNTIME!")
                            swapped = True
                            break
                            
                        time.sleep(0.2)  # Tiny throttle for readability
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error during drift streaming: {e}")

    if not swapped:
        print("\nSimulation ended. Hot-swap was not triggered. Check MLOps Worker logs for details.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        generate_reference_data_and_baseline()
    else:
        generate_reference_data_and_baseline()
        run_simulation()

import pytest
import json
import numpy as np
from backend.app import app
from backend.dataset.preprocessing import encode_array_to_base64

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"

def test_dataset_info_endpoint(client):
    res = client.get("/api/dataset/info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["num_classes"] == 10

def test_dataset_samples_endpoint(client):
    res = client.get("/api/dataset/samples?per_class=1")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["samples"]) >= 10

def test_train_and_status(client):
    # Train Perceptron on small subset for fast test
    payload = {
        "algorithm": "perceptron",
        "dataset_size": 200,
        "epochs": 2,
        "learning_rate": 0.01,
        "random_seed": 42
    }
    res = client.post("/api/train", json=payload)
    assert res.status_code == 202
    data = res.get_json()
    assert data["success"] is True
    job_id = data["job_id"]

    # Poll status
    status_res = client.get(f"/api/train/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.get_json()
    assert status_data["success"] is True
    assert status_data["job"]["algorithm"] == "perceptron"

def test_compare_and_export(client):
    comp_res = client.get("/api/compare")
    assert comp_res.status_code == 200
    comp_data = comp_res.get_json()
    assert comp_data["success"] is True
    assert "models" in comp_data["data"]
    assert "comparison_table" in comp_data["data"]

    csv_res = client.get("/api/export/results")
    assert csv_res.status_code == 200
    assert "Learning Rule" in csv_res.get_data(as_text=True)

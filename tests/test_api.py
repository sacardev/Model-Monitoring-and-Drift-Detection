import os
os.environ["DRIFT_SCHEDULER_ENABLED"] = "false"   # keep the background job out of test runs

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_RECORD = {
    "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 840.0,
    "gender": "Male", "Contract": "Month-to-month",
    "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_one_result_per_record():
    response = client.post("/predict", json={"records": [SAMPLE_RECORD]})
    assert response.status_code == 200
    assert response.json()["results"][0]["prediction"] in (0, 1)

def test_ingest_logs_records():
    response = client.post("/ingest", json={"records": [SAMPLE_RECORD] * 5})
    assert response.status_code == 201
    assert response.json()["ingested"] == 5
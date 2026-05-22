import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_valid_payload():
    """Inference returns a valid numeric prediction for a correct request."""
    with open("artifacts/example_payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0

    pred = data["predictions"][0]["duration_min_predicted"]
    assert isinstance(pred, (int, float))
    assert pred > 0

def test_predict_missing_records_field():
    """API should reject payloads missing required fields."""
    bad_payload = {
        # missing "records"
    }

    response = client.post("/predict", json=bad_payload)

    assert response.status_code in (400, 422)

def test_predict_store_and_fwd_flag_edge_case():
    """Inference handles categorical edge cases like lowercase flags."""
    with open("artifacts/example_payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Modify categorical value to a common edge case
    payload["records"][0]["store_and_fwd_flag"] = "n"

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    pred = response.json()["predictions"][0]["duration_min_predicted"]
    assert pred > 0



from src.api import app


def test_predict_endpoint():
    """Test the /predict endpoint."""

    client = app.test_client()

    payload = {
        "feature1": 10,
        "feature2": 20,
        "feature3": "value",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data


def test_predict_invalid_request():
    """Test /predict with an empty request."""

    client = app.test_client()

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code in [400, 422]


def test_predict_method_not_allowed():
    """GET should not be allowed on /predict."""

    client = app.test_client()

    response = client.get("/predict")

    assert response.status_code == 405


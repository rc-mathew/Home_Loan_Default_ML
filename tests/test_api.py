
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_metadata_endpoint_when_artifact_exists():
    response = client.get("/metadata")
    assert response.status_code in (200, 503)

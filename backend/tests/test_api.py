import pytest
from app.main import app, inference_service
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    from app.core import database

    database.engine = database.create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    database.SessionLocal.configure(bind=database.engine)
    original_load = inference_service.load
    inference_service.load = lambda: setattr(inference_service, "model", object())
    with TestClient(app) as test_client:
        yield test_client
    inference_service.load = original_load
    inference_service.model = None


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    assert response.json()["classes"] == [{"id": 0, "name": "person"}]

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Italian LLM API"


def test_tools_list():
    resp = client.get("/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "web_search" in data
    assert "calculator" in data
    assert "wikipedia" in data


def test_chat_no_model():
    resp = client.post("/chat", json={"message": "Ciao", "use_rag": False, "use_web_search": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "conversation_id" in data


def test_chat_calculator():
    resp = client.post("/chat", json={"message": "Quanto fa 128 per 56?", "use_rag": False, "use_web_search": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "calculator" in data["tools_used"]


def test_chat_datetime():
    resp = client.post("/chat", json={"message": "Che ore sono?", "use_rag": False, "use_web_search": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "datetime" in data["tools_used"]

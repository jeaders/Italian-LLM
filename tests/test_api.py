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


def test_budget_status():
    resp = client.get("/budget/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "monthly_budget" in data
    assert "spent" in data
    assert "remaining" in data
    assert data["monthly_budget"] == 100.0


def test_budget_add_expense():
    resp = client.post("/budget/expenses", json={"amount": 5.5, "category": "cibo", "description": "pasta"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["expense"]["amount"] == 5.5
    assert data["expense"]["category"] == "cibo"


def test_budget_status_after_expense():
    resp = client.get("/budget/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["spent"] >= 5.5
    assert data["remaining"] <= 94.5


def test_budget_advice():
    resp = client.get("/budget/advice")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "tip" in data
    assert "actions" in data


def test_budget_reset_requires_confirm():
    resp = client.post("/budget/reset", json={"confirm": False})
    assert resp.status_code == 400


def test_budget_reset():
    resp = client.post("/budget/expenses", json={"amount": 1.0, "category": "test", "description": "reset-test"})
    assert resp.status_code == 200
    resp = client.post("/budget/reset", json={"confirm": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reset"
    status = client.get("/budget/status").json()
    assert status["spent"] == 0.0
    assert status["remaining"] == 100.0

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import budget_tracker

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_budget_state():
    """Every test starts from a clean budget store (the JSON file is shared state)."""
    if budget_tracker.BUDGET_FILE.exists():
        budget_tracker.BUDGET_FILE.unlink()
    yield
    if budget_tracker.BUDGET_FILE.exists():
        budget_tracker.BUDGET_FILE.unlink()


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
    client.post("/budget/expenses", json={"amount": 5.5, "category": "cibo", "description": "pasta"})
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
    assert data["tip_source"] in ("ai", "rules")


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


def test_budget_reset_clears_income():
    """Regression: reset used to keep previous income, inflating the remaining budget."""
    resp = client.post("/budget/income", json={"amount": 30.0, "source": "test"})
    assert resp.status_code == 200
    resp = client.post("/budget/reset", json={"confirm": True})
    assert resp.status_code == 200
    status = client.get("/budget/status").json()
    assert status["income"] == 0.0
    assert status["remaining"] == 100.0


def test_month_rollover_clears_previous_month(tmp_path, monkeypatch):
    """Regression: month rollover used to keep last month's income and expenses."""
    state_file = tmp_path / "budget_tracker.json"
    state_file.write_text(json.dumps({
        "monthly_budget": 100.0,
        "current_month": "2026-01",
        "expenses": [{"amount": 10.0, "category": "cibo", "description": "", "date": "2026-01-01T00:00:00", "timestamp": 0}],
        "income": [{"amount": 50.0, "source": "test", "date": "", "timestamp": 0}],
    }), encoding="utf-8")
    monkeypatch.setattr(budget_tracker, "BUDGET_FILE", state_file)

    status = budget_tracker.get_status()
    assert status["spent"] == 0.0
    assert status["income"] == 0.0
    assert status["remaining"] == 100.0

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "2026-01" in saved["history"]
    assert saved["history"]["2026-01"]["spent"] == 10.0


def test_action_blocked_when_deactivated():
    """Regression: actions used to work after Pippo's deactivation."""
    client.post("/budget/expenses", json={"amount": 130.0, "category": "test", "description": "over-budget"})
    status = client.get("/budget/status").json()
    assert status["deactivated"] is True

    resp = client.post("/budget/actions/microtask")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False


def test_pippo_state_for_prompt():
    state = budget_tracker.get_pippo_state()
    assert state["monthly_budget"] == 100.0
    assert state["remaining"] == 100.0
    assert state["vital_status"] == "alive"
    assert state["days_left"] >= 0


def test_tool_calculator_italian_words():
    from api.services.tools import tool_calculator
    assert tool_calculator("128 per 56") == "7168"
    assert tool_calculator("2 + 2") == "4"
    assert tool_calculator("10 diviso 4") == "2.5"
    assert "Errore" in tool_calculator("__import__('os').system('ls')")


def test_wikipedia_tool_has_no_nameerror():
    """Regression: tool_wikipedia/tool_weather/tool_news referenced httpx without importing it."""
    from api.services.tools import tool_wikipedia
    result = tool_wikipedia("Pippo")
    assert "name 'httpx' is not defined" not in result

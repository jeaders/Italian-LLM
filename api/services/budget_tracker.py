import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/budget")
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUDGET_FILE = DATA_DIR / "budget_tracker.json"


def load_budget() -> Dict[str, Any]:
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "monthly_budget": 100.0,
        "current_month": datetime.now().strftime("%Y-%m"),
        "expenses": [],
        "savings_tips": []
    }


def save_budget(budget: Dict[str, Any]) -> None:
    with open(BUDGET_FILE, 'w', encoding='utf-8') as f:
        json.dump(budget, f, ensure_ascii=False, indent=2)


def get_current_month_budget() -> Dict[str, Any]:
    budget = load_budget()
    now = datetime.now()
    current_month = now.strftime("%Y-%m")
    if budget.get("current_month") != current_month:
        budget["current_month"] = current_month
        budget["expenses"] = []
        save_budget(budget)
    return budget


def add_expense(amount: float, category: str, description: str = "") -> Dict[str, Any]:
    budget = get_current_month_budget()
    expense = {
        "amount": round(amount, 2),
        "category": category,
        "description": description,
        "date": datetime.now().isoformat(),
        "timestamp": datetime.now().timestamp()
    }
    budget["expenses"].append(expense)
    save_budget(budget)
    return expense


def get_spent() -> float:
    budget = get_current_month_budget()
    return round(sum(e["amount"] for e in budget["expenses"]), 2)


def get_remaining() -> float:
    budget = get_current_month_budget()
    spent = get_spent()
    return round(budget["monthly_budget"] - spent, 2)


def get_expenses_by_category() -> Dict[str, float]:
    budget = get_current_month_budget()
    by_cat = defaultdict(float)
    for e in budget["expenses"]:
        by_cat[e["category"]] = round(by_cat[e["category"]] + e["amount"], 2)
    return dict(by_cat)


def get_status() -> Dict[str, Any]:
    budget = get_current_month_budget()
    spent = get_spent()
    remaining = get_remaining()
    by_cat = get_expenses_by_category()
    percentage = round((spent / budget["monthly_budget"]) * 100, 1) if budget["monthly_budget"] > 0 else 0.0
    alert = None
    if remaining < 0:
        alert = f"Attenzione: hai superato il budget di {abs(remaining):.2f} euro!"
    elif percentage > 75:
        alert = f"Attenzione: hai speso il {percentage}% del budget, ti rimangono solo {remaining:.2f} euro."
    return {
        "monthly_budget": budget["monthly_budget"],
        "spent": spent,
        "remaining": remaining,
        "percentage_used": percentage,
        "by_category": by_cat,
        "alert": alert,
        "expenses_count": len(budget["expenses"])
    }


def reset_month() -> Dict[str, Any]:
    budget = load_budget()
    budget["current_month"] = datetime.now().strftime("%Y-%m")
    budget["expenses"] = []
    save_budget(budget)
    return budget


def generate_tip(remaining: float, by_category: Dict[str, float], model_fn=None) -> str:
    if remaining < 0:
        return "Hai superato il budget. Taglia subito: cibo fuori casa, abbonamenti, trasporti non essenziali. Rivolgiti ai servizi sociali e ai banchi alimentari."
    if remaining < 10:
        return "Budget critico. Acquista solo alimenti base: pasta, riso, legumi, uova. Usa i banchi alimentari e le mense sociali. Evita qualsiasi spesa non essenziale."
    if remaining < 25:
        return "Budget sotto pressione. Pianifica i pasti settimanali, evita sprechi, compra solo in discount, e sospendi tutti gli abbonamenti e servizi a pagamento."
    if by_category.get("cibo", 0) > 35:
        return "Stai spendendo troppo per cibo. Passa a discount, compra all'ingrosso, cucina sempre a casa e evita takeaway e snack."
    if by_category.get("trasporti", 0) > 15:
        return "Riduci i trasporti: usa bici, mezzi pubblici o car sharing. Cammina per tragitti brevi e organizza passaggi con colleghi."
    return "Continua così: monitora ogni spesa, cerca offerte, e ricorda che ogni euro risparmiato ti aiuta a raggiungere la fine del mese."


def get_survival_advice(model_fn=None) -> Dict[str, Any]:
    status = get_status()
    tip = generate_tip(status["remaining"], status["by_category"], model_fn)
    return {
        "status": status,
        "tip": tip,
        "actions": [
            "Pianifica i pasti della settimana e compra solo il necessario",
            "Usa i banchi alimentari e le mense sociali",
            "Sospendi tutti gli abbonamenti e servizi non essenziali",
            "Riduci al minimo le spese di trasporto",
            "Rivolgiti ai servizi sociali del tuo comune per aiuti economici"
        ]
    }

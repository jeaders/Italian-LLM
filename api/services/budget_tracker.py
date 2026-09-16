import json
import logging
from datetime import datetime, timedelta
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
    spent_ratio = 1.0 - (remaining / 100.0) if remaining >= 0 else 2.0
    if remaining < 0:
        return "Budget IN ROSSO. Ferma TUTTE le spese non vitali. Sopravvivi con pasta/riso/legumi/pane surgelato dai discount e banchi alimentari. Chiedi subito aiuto a servizi sociali e mense. Ogni euro deve servire a restare in vita."
    if remaining < 10:
        return "Sopravvivenza critica. Solo alimenti base: pasta, riso, legumi secchi, uova, pane surgelato. Acquista nei discount più vicini, evita prodotti marca e biologici. Cucina a casa, niente takeaway, niente caffè fuori, niente snack. Chiedi supporto ai banchi alimentari e alle mense sociali."
    if remaining < 25:
        return "Modalità sopravvivenza attiva. Budget giornaliero: massimo 0,80€ per il cibo. Pasta e legumi sono i tuoi alleati. Acquista all'ingrosso o in family size quando possibile. Cancella ogni abbonamento, usa mezzi pubblici o bici, non spendere per svago finché non sei in pari."
    if by_category.get("cibo", 0) > 35:
        return "Con 100€ al mese il cibo non deve superare i 35€ totali. Compra pasta, riso, legumi, uova e verdure in discount. Evita takeaway, snack, bibite e caffè al bar. Cucina sempre e prepara i pasti in grandi quantità per risparmiare tempo e denaro."
    if by_category.get("trasporti", 0) > 15:
        return "Trasporti oltre il budget di sopravvivenza. Usa la bici, cammina per tragitti brevi, preferisci mezzi pubblici con abbonamento mensile conveniente. Evita taxi, ride-hailing e ogni spostamento non essenziale."
    if spent_ratio > 0.8:
        return "Siamo all'80% del budget. Attiva la modalità sopravvivenza: niente spese non essenziali, priorità assoluta a cibo base, bollette minime e salute. Cerca coupon, sconti, community di scambio e aiuti territoriali."
    return "Stai gestendo il budget di 100€/mese. Continua a monitorare ogni spesa: compra in discount, cucina sempre a casa, evita sprechi, sospendi abbonamenti non essenziali e ricorda che ogni euro risparmiato è un passo in più verso la sopravvivenza."


def get_daily_budget() -> float:
    now = datetime.now()
    days_in_month = (now.replace(day=28) + timedelta(days=4)).day
    remaining = get_remaining()
    return round(max(remaining / max(days_in_month, 1), 0.0), 2)


def get_survival_goals() -> Dict[str, Any]:
    budget = get_current_month_budget()
    remaining = get_remaining()
    by_cat = get_expenses_by_category()
    food_spent = by_cat.get("cibo", 0.0)
    transport_spent = by_cat.get("trasporti", 0.0)
    bills_spent = by_cat.get("bollette", 0.0)

    food_limit = 35.0
    transport_limit = 15.0
    daily_food_budget = round(food_limit / 30, 2)

    goals = [
        {
            "label": "Cibo (max 35€/mese)",
            "spent": food_spent,
            "limit": food_limit,
            "unit": "€",
            "daily_budget": daily_food_budget,
            "status": "ok" if food_spent <= food_limit else "over",
        },
        {
            "label": "Trasporti (max 15€/mese)",
            "spent": transport_spent,
            "limit": transport_limit,
            "unit": "€",
            "status": "ok" if transport_spent <= transport_limit else "over",
        },
        {
            "label": "Bollette essenziali",
            "spent": bills_spent,
            "limit": None,
            "unit": "€",
            "status": "ok",
        },
        {
            "label": "Rimanenti totali",
            "spent": None,
            "limit": budget["monthly_budget"],
            "unit": "€",
            "value": remaining,
            "status": "ok" if remaining >= 0 else "over",
        },
    ]
    return {"daily_budget": get_daily_budget(), "goals": goals}


def get_survival_advice(model_fn=None) -> Dict[str, Any]:
    status = get_status()
    tip = generate_tip(status["remaining"], status["by_category"], model_fn)
    goals = get_survival_goals()
    survival_actions = [
        "Compra pasta, riso, legumi e uova in discount: sono la base della sopravvivenza con 100€/mese",
        "Pianifica i pasti settimanali e cucina sempre a casa: evita takeaway, caffè fuori e snack",
        "Usa i banchi alimentari e le mense sociali: non vergognarti, sono risorse pubbliche",
        "Cancella ogni abbonamento non essenziale (streaming, app, palestre): risparmia decine di euro",
        "Muoviti a piedi o in bici: elimina i costi di trasporto non essenziali",
        "Cerca coupon, sconti e offerte: supermercati e siti di deal possono aiutare",
        "Rivolgiti ai servizi sociali del tuo comune per aiuti economici e buoni spesa",
        "Scambia beni e servizi con la community: il baratto è libero e gratuito",
    ]
    return {
        "status": status,
        "tip": tip,
        "actions": survival_actions,
        "goals": goals,
    }

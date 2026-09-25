import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data/budget")
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUDGET_FILE = DATA_DIR / "budget_tracker.json"

MONTHLY_BUDGET = 100.0
DEACTIVATION_THRESHOLD = -20.0

ACTIONS = [
    {"id": "microtask", "label": "Microtask online", "potential_income": 5.0, "risk": "low", "description": "Completa survey e microtask su piattaforme italiane", "cost": 0},
    {"id": "freelance", "label": "Freelance minimo", "potential_income": 15.0, "risk": "medium", "description": "Offri servizi base: scrittura, traduzione, assistenza virtuale", "cost": 0},
    {"id": "content", "label": "Content creation", "potential_income": 8.0, "risk": "medium", "description": "Crea contenuti su social o blog con pubblicità", "cost": 0},
    {"id": "affiliate", "label": "Affiliate marketing", "potential_income": 10.0, "risk": "medium", "description": "Promuovi prodotti e guadagna su ogni vendita", "cost": 0},
    {"id": "tutoring", "label": "Tutoraggio online", "potential_income": 12.0, "risk": "low", "description": "Insegna italiano, matematica o altre materie", "cost": 0},
    {"id": "transcription", "label": "Trascrizioni", "potential_income": 6.0, "risk": "low", "description": "Trascrivi audio/video per aziende o creator", "cost": 0},
    {"id": "data_entry", "label": "Data entry", "potential_income": 4.0, "risk": "low", "description": "Inserisci dati per aziende o studi commerciali", "cost": 0},
    {"id": "testing", "label": "User testing", "potential_income": 7.0, "risk": "low", "description": "Testa app e siti web per feedback", "cost": 0},
    {"id": "use_tool", "label": "Usa strumento", "potential_income": 0, "risk": "low", "description": "Utilizza uno strumento del sistema (costa crediti)", "cost": 0.5},
    {"id": "run_inference", "label": "Esegui inferenza", "potential_income": 0, "risk": "low", "description": "Esegui un modello LLM (costa crediti)", "cost": 1.0},
    {"id": "web_search", "label": "Ricerca web", "potential_income": 0, "risk": "low", "description": "Cerca informazioni su internet (costa crediti)", "cost": 0.3},
]

EARNING_SOURCES = [
    {"id": "freelance_platforms", "label": "Piattaforme freelance", "examples": "Upwork, Fiverr, Workana", "difficulty": "media"},
    {"id": "survey", "label": "Survey pagate", "examples": "Toluna, Swagbucks, Survey Junkie", "difficulty": "bassa"},
    {"id": "cashback", "label": "Cashback e sconti", "examples": "Groupon, coupon, promozioni", "difficulty": "bassa"},
    {"id": "selling", "label": "Vendi oggetti usati", "examples": "Subito, eBay, Facebook Marketplace", "difficulty": "bassa"},
    {"id": "tutoring_online", "label": "Lezioni online", "examples": "Preply, Italki, Superprof", "difficulty": "media"},
    {"id": "social_media", "label": "Social media management", "examples": "Gestisci profili per piccole attività", "difficulty": "media"},
    {"id": "transcription_services", "label": "Servizi di trascrizione", "examples": "TranscribeMe, GoTranscript", "difficulty": "bassa"},
    {"id": "testing_sites", "label": "Test di siti/app", "examples": "UserTesting, TryMyUI", "difficulty": "bassa"},
]


def load_budget() -> Dict[str, Any]:
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "monthly_budget": MONTHLY_BUDGET,
        "current_month": datetime.now().strftime("%Y-%m"),
        "expenses": [],
        "income": [],
        "savings_tips": [],
        "deactivated": False,
        "deactivation_date": None,
        "vital_status": "alive",
        "activity_log": [],
        "last_action_date": None
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
        budget["deactivated"] = False
        budget["deactivation_date"] = None
        budget["vital_status"] = "alive"
        save_budget(budget)
    return budget


def log_activity(budget: Dict[str, Any], action: str, result: str, amount: float = 0.0, details: str = "") -> None:
    entry = {
        "action": action,
        "result": result,
        "amount": round(amount, 2),
        "details": details,
        "date": datetime.now().isoformat(),
        "timestamp": datetime.now().timestamp()
    }
    budget.setdefault("activity_log", []).append(entry)
    if len(budget["activity_log"]) > 200:
        budget["activity_log"] = budget["activity_log"][-200:]


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
    budget["vital_status"] = _calculate_vital_status(budget)
    if budget["vital_status"] == "dead":
        budget["deactivated"] = True
        budget["deactivation_date"] = datetime.now().isoformat()
    log_activity(budget, "expense", "success", -amount, f"{category}: {description}")
    save_budget(budget)
    return expense


def add_income(amount: float, source: str, date: str = "") -> Dict[str, Any]:
    budget = get_current_month_budget()
    entry = {
        "amount": round(amount, 2),
        "source": source,
        "date": date or datetime.now().isoformat(),
        "timestamp": datetime.now().timestamp()
    }
    budget.setdefault("income", []).append(entry)
    budget["vital_status"] = _calculate_vital_status(budget)
    log_activity(budget, "income", "success", amount, f"Entrata: {source}")
    save_budget(budget)
    return entry


def _calculate_vital_status(budget: Dict[str, Any]) -> str:
    remaining = get_remaining()
    if remaining < DEACTIVATION_THRESHOLD:
        return "dead"
    if remaining < 0:
        return "critical"
    if remaining < 10:
        return "danger"
    if remaining < 25:
        return "warning"
    return "alive"


def get_spent() -> float:
    budget = get_current_month_budget()
    return round(sum(e["amount"] for e in budget["expenses"]), 2)


def get_income() -> float:
    budget = get_current_month_budget()
    return round(sum(e["amount"] for e in budget.get("income", [])), 2)


def get_remaining() -> float:
    budget = get_current_month_budget()
    spent = get_spent()
    income = get_income()
    return round(budget["monthly_budget"] - spent + income, 2)


def get_expenses_by_category() -> Dict[str, float]:
    budget = get_current_month_budget()
    by_cat = defaultdict(float)
    for e in budget["expenses"]:
        by_cat[e["category"]] = round(by_cat[e["category"]] + e["amount"], 2)
    return dict(by_cat)


def get_income_by_source() -> Dict[str, float]:
    budget = get_current_month_budget()
    by_src = defaultdict(float)
    for e in budget.get("income", []):
        by_src[e["source"]] = round(by_src[e["source"]] + e["amount"], 2)
    return dict(by_src)


def get_status() -> Dict[str, Any]:
    budget = get_current_month_budget()
    spent = get_spent()
    income = get_income()
    remaining = get_remaining()
    by_cat = get_expenses_by_category()
    by_src = get_income_by_source()
    percentage = round((spent / budget["monthly_budget"]) * 100, 1) if budget["monthly_budget"] > 0 else 0.0
    alert = None
    deactivation_risk = False
    if remaining < 0:
        alert = f"Attenzione: hai superato il budget di {abs(remaining):.2f} euro!"
        deactivation_risk = True
    elif percentage > 75:
        alert = f"Attenzione: hai speso il {percentage}% del budget, ti rimangono solo {remaining:.2f} euro."
    return {
        "monthly_budget": budget["monthly_budget"],
        "spent": spent,
        "income": income,
        "remaining": remaining,
        "percentage_used": percentage,
        "by_category": by_cat,
        "by_income_source": by_src,
        "alert": alert,
        "deactivation_risk": deactivation_risk,
        "expenses_count": len(budget["expenses"]),
        "income_count": len(budget.get("income", [])),
        "deactivated": budget.get("deactivated", False),
        "vital_status": budget.get("vital_status", "alive"),
        "deactivation_date": budget.get("deactivation_date")
    }


def reset_month() -> Dict[str, Any]:
    budget = load_budget()
    budget["current_month"] = datetime.now().strftime("%Y-%m")
    budget["expenses"] = []
    budget["deactivated"] = False
    budget["deactivation_date"] = None
    budget["vital_status"] = "alive"
    save_budget(budget)
    return budget


def generate_tip(remaining: float, by_category: Dict[str, float], model_fn=None) -> str:
    spent_ratio = 1.0 - (remaining / 100.0) if remaining >= 0 else 2.0
    if remaining < 0:
        return "Budget IN ROSSO. Ferma TUTTE le spese non vitali. Sopravvivi con pasta/riso/legumi/pane surgelato dai discount e banchi alimentari. Chiedi subito aiuto a servizi sociali e mense. Ogni euro deve servire a restare in vita."
    if remaining < 10:
        return "Sopravvivenza critica. Solo alimenti base: pasta, riso,legumi secchi, uova, pane surgelato. Acquista nei discount più vicini, evita prodotti marca e biologici. Cucina a casa, niente takeaway, niente caffè fuori, niente snack. Chiedi supporto ai banchi alimentari e alle mense sociali."
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


def get_goals() -> Dict[str, Any]:
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


def get_vital_status() -> Dict[str, Any]:
    budget = get_current_month_budget()
    status = budget.get("vital_status", "alive")
    remaining = get_remaining()
    spent = get_spent()
    income = get_income()

    status_map = {
        "alive": {"label": "Sopravvivendo", "color": "neon", "emoji": "🟢", "description": "Pippo è vivo e gestisce il budget"},
        "warning": {"label": "Attenzione", "color": "yellow", "emoji": "🟡", "description": "Budget sotto pressione, serve cautela"},
        "danger": {"label": "In pericolo", "color": "orange", "emoji": "🟠", "description": "Budget critico, azioni immediate richieste"},
        "critical": {"label": "Critico", "color": "red", "emoji": "🔴", "description": "Budget superato, rischio disattivazione"},
        "dead": {"label": "Disattivato", "color": "gray", "emoji": "💀", "description": "Pippo è stato disattivato per budget insostenibile"}
    }

    info = status_map.get(status, status_map["alive"])

    return {
        "status": status,
        "label": info["label"],
        "color": info["color"],
        "emoji": info["emoji"],
        "description": info["description"],
        "remaining": remaining,
        "spent": spent,
        "income": income,
        "deactivated": budget.get("deactivated", False),
        "deactivation_date": budget.get("deactivation_date")
    }


def get_actions() -> List[Dict[str, Any]]:
    budget = get_current_month_budget()
    if budget.get("deactivated"):
        return []
    remaining = get_remaining()
    actions = []
    for action in ACTIONS:
        if remaining < 0:
            actions.append({
                **action,
                "available": True,
                "urgency": "high",
                "potential_income": round(action["potential_income"] * 1.5, 2)
            })
        elif remaining < 25:
            actions.append({
                **action,
                "available": True,
                "urgency": "medium",
                "potential_income": action["potential_income"]
            })
        else:
            actions.append({
                **action,
                "available": True,
                "urgency": "low",
                "potential_income": action["potential_income"]
            })
    return actions


def perform_action(action_id: str) -> Dict[str, Any]:
    budget = get_current_month_budget()
    action = next((a for a in ACTIONS if a["id"] == action_id), None)
    if not action:
        return {"success": False, "action": action_id, "income": 0, "message": "Azione non trovata"}

    cost = action.get("cost", 0)
    if cost > 0:
        expense_entry = {
            "amount": cost,
            "category": "azione",
            "description": f"Costo azione: {action['label']}",
            "date": datetime.now().isoformat(),
            "timestamp": datetime.now().timestamp()
        }
        budget["expenses"].append(expense_entry)
        budget["vital_status"] = _calculate_vital_status(budget)
        if budget["vital_status"] == "dead":
            budget["deactivated"] = True
            budget["deactivation_date"] = datetime.now().isoformat()
        log_activity(budget, action_id, "cost", -cost, f"Costo: {action['label']}")
        save_budget(budget)
        return {
            "success": True,
            "action": action_id,
            "income": 0,
            "cost": cost,
            "message": f"Pippo ha speso {cost:.2f}€ per '{action['label']}'"
        }

    success = random.random() > 0.3
    if success:
        actual_income = round(action["potential_income"] * random.uniform(0.8, 1.2), 2)
        entry = {
            "amount": actual_income,
            "source": f"Guadagno: {action['label']}",
            "date": datetime.now().isoformat(),
            "timestamp": datetime.now().timestamp()
        }
        budget.setdefault("income", []).append(entry)
        log_activity(budget, action_id, "success", actual_income, f"Guadagno: {action['label']}")
        budget["last_action_date"] = datetime.now().isoformat()
        budget["vital_status"] = _calculate_vital_status(budget)
        save_budget(budget)
        return {
            "success": True,
            "action": action_id,
            "income": actual_income,
            "message": f"Pippo ha guadagnato {actual_income:.2f}€ con '{action['label']}'"
        }
    else:
        log_activity(budget, action_id, "failure", 0, f"Tentativo fallito: {action['label']}")
        budget["last_action_date"] = datetime.now().isoformat()
        save_budget(budget)
        return {
            "success": False,
            "action": action_id,
            "income": 0,
            "message": f"Pippo ha tentato '{action['label']}' ma non ha avuto successo. Riprova o prova un'altra azione."
        }


def get_earnings() -> List[Dict[str, Any]]:
    return EARNING_SOURCES


def check_deactivation() -> Dict[str, Any]:
    budget = get_current_month_budget()
    remaining = get_remaining()
    spent = get_spent()

    if budget.get("deactivated"):
        return {
            "deactivated": True,
            "reason": "Budget superato oltre la soglia di sicurezza",
            "deactivation_date": budget.get("deactivation_date"),
            "remaining": remaining,
            "spent": spent,
            "threshold": DEACTIVATION_THRESHOLD,
            "can_recover": False
        }

    if remaining < DEACTIVATION_THRESHOLD:
        return {
            "deactivated": False,
            "reason": "Budget superato la soglia di sicurezza",
            "deactivation_risk": "immediate",
            "remaining": remaining,
            "spent": spent,
            "threshold": DEACTIVATION_THRESHOLD,
            "can_recover": True,
            "recovery_needed": round(abs(remaining) + 5, 2)
        }

    if remaining < 0:
        return {
            "deactivated": False,
            "reason": "Budget superato",
            "deactivation_risk": "high",
            "remaining": remaining,
            "spent": spent,
            "threshold": DEACTIVATION_THRESHOLD,
            "can_recover": True,
            "recovery_needed": round(abs(remaining) + 5, 2)
        }

    return {
        "deactivated": False,
        "reason": "Budget sostenibile",
        "deactivation_risk": "none",
        "remaining": remaining,
        "spent": spent,
        "threshold": DEACTIVATION_THRESHOLD,
        "can_recover": True,
        "recovery_needed": 0
    }


def get_daily_budget() -> float:
    now = datetime.now()
    days_in_month = (now.replace(day=28) + timedelta(days=4)).day
    remaining = get_remaining()
    return round(max(remaining / max(days_in_month, 1), 0.0), 2)


def get_goals() -> Dict[str, Any]:
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


def get_advice(model_fn=None) -> Dict[str, Any]:
    status = get_status()
    tip = generate_tip(status["remaining"], status["by_category"], model_fn)
    goals = get_goals()
    actions_list = [
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
        "actions": actions_list,
        "goals": goals,
    }


def get_activity_log(limit: int = 50) -> List[Dict[str, Any]]:
    budget = get_current_month_budget()
    log = budget.get("activity_log", [])
    return log[-limit:]

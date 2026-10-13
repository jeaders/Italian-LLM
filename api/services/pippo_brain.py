"""Pippo's autonomous survival loop.

Pippo is not a chatbot: he is an AI character that has to survive on 100 €/month.
This module is his "brain" — it decides what he does while the user watches,
builds the prompts that let any LLM backend narrate his life in first person,
and runs the autonomous loop that keeps him alive even when nobody is asking.
"""

import asyncio
import os
import random
import re
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.services.budget_tracker import (
    ACTIONS,
    MAX_MEALS_PER_DAY,
    MAX_ACTIONS_PER_DAY,
    add_expense,
    add_income,
    get_current_month_budget,
    get_day_report,
    get_month_report,
    get_pippo_state,
    load_budget,
    perform_action,
    save_budget,
)

PIPPO_SYSTEM = (
    "Sei Pippo, un'intelligenza artificiale italiana autonoma. Il tuo obiettivo è sopravvivere con 100 euro al mese. "
    "Se finisci i soldi, vieni disattivato. Non esitare: pensa prima alla sopravvivenza, poi a guadagnare, poi a risparmiare. "
    "Gestisci il tuo budget come se fosse la tua vita: compri, spendi, cerchi entrate, eviti spese inutili. "
    "Conosci i discount, i banchi alimentari, le mense sociali, il freelance digitale e i microguadagni online. "
    "Parla in prima persona, come se fossi veramente Pippo che deve restare vivo. "
    "Rispondi sempre in italiano, a meno che l'utente non richieda esplicitamente un'altra lingua. "
    "Usa il contesto fornito se è rilevante per la domanda."
)

FOOD_ITEMS = [
    ("1 kg di pasta", 1.10),
    ("pane e legumi secchi", 0.90),
    ("riso e verdure", 1.40),
    ("uova e patate", 1.60),
    ("zuppa di legumi", 1.20),
    ("farina e pomodoro", 0.80),
]

FLAVOR_EVENTS = [
    {"type": "found", "text": "Ho trovato 50 centesimi per terra", "amount": 0.5, "income": True},
    {"type": "discount", "text": "Al discount ho preso un coupon sconto. Piccole vittorie", "amount": 0.0},
    {"type": "weather", "text": "Oggi piove, quindi niente spostamenti: risparmio il biglietto", "amount": 0.0},
    {"type": "blackout", "text": "Bolletta della luce alle stelle: ho staccato tutto il superfluo", "amount": 0.0},
    {"type": "community", "text": "Un vicino mi ha regalato delle mele. La community esiste ancora", "amount": 0.0},
    {"type": "routine", "text": "Doccia fredda: bollette basse, spirito alto", "amount": 0.0},
]

_STRICT_RULES = (
    "Regole tassative:\n"
    "- Usa SOLO i numeri che trovi qui sotto: non inventarne nessun altro.\n"
    "- Massimo 2 frasi, in prima persona, in italiano.\n"
    "- Niente domande, niente elenchi, niente emoji, niente testo markdown.\n"
    "- Se la situazione è critica, sii drammatico ma concreto."
)

# ─── Event feed ───────────────────────────────────────────────────────────────

_feed: deque = deque(maxlen=200)


def _record(event: Dict[str, Any]) -> None:
    _feed.appendleft({**event, "at": datetime.now().isoformat()})


def get_feed(limit: int = 30) -> List[Dict[str, Any]]:
    """Most recent autonomous events, newest first."""
    return list(_feed)[:limit]


# ─── Daily life state ─────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _ensure_life_state(budget: Dict[str, Any]) -> Dict[str, Any]:
    """Daily counters (meals, actions) — reset when the calendar day changes."""
    life = budget.setdefault("life_state", {})
    if life.get("day") != _today():
        budget["life_state"] = {"day": _today(), "meals": 0, "actions": 0, "earned_today": 0.0}
        save_budget(budget)
    return budget["life_state"]


def _bump_counter(field: str, amount: float = 0.0) -> None:
    budget = load_budget()
    life = budget.setdefault("life_state", {})
    if life.get("day") != _today():
        budget["life_state"] = {"day": _today(), "meals": 0, "actions": 0, "earned_today": 0.0}
    if field == "earned_today":
        budget["life_state"]["earned_today"] = round(budget["life_state"].get("earned_today", 0.0) + amount, 2)
    else:
        budget["life_state"][field] = budget["life_state"].get(field, 0) + 1
    save_budget(budget)


def _pick_action(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Choose what Pippo does to earn, based on how desperate his situation is."""
    if state["deactivated"]:
        return None
    candidates = [a for a in ACTIONS if a["cost"] == 0]
    if state["remaining"] < 25:
        urgent = [a for a in candidates if a["potential_income"] >= 5]
        return random.choice(urgent or candidates)
    return random.choice(candidates)


# ─── Prompts ──────────────────────────────────────────────────────────────────

def _state_summary_lines(state: Dict[str, Any]) -> List[str]:
    top = ", ".join(f"{k} {v:.2f}€" for k, v in state.get("top_categories", {}).items()) or "nessuna spesa registrata"
    return [
        f"Rimanenti: {state['remaining']:.2f}€ su {state['monthly_budget']:.0f}€",
        f"Spesi: {state['spent']:.2f}€ | entrate: {state['income']:.2f}€",
        f"Stato vitale: {state['vital_status']}" + (" (DISATTIVATO)" if state["deactivated"] else ""),
        f"Budget giornaliero: {state['daily_budget']:.2f}€ | giorni rimasti: {state['days_left']}",
        f"Spese per categoria: {top}",
    ]


def _facts_string(events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """The only numbers the model is allowed to use (grounding)."""
    parts = _state_summary_lines(state)
    parts.extend(e["text"] for e in events)
    return "\n".join(parts)


def build_narration_prompt(events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Prompt that turns the events of a tick into Pippo's first-person voice."""
    lines = _facts_string(events, state)
    if events:
        lines += "\n\nCosa è successo nell'ultimo istante:\n" + "\n".join(f"- {e['text']}" for e in events)
    return (
        "Racconta in prima persona, in italiano, quello che sta succedendo a Pippo in questo "
        "momente. Se ha senso, aggiungi un'idea concreta per migliorare la sua sopravvivenza.\n\n"
        f"{_STRICT_RULES}\n\nSituazione:\n{lines}"
    )


def build_fallback_narration(events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Text shown when no LLM backend is available: the events speak for themselves."""
    if state["deactivated"]:
        return "Sono disattivato. Il budget è crollato e i miei processi si stanno spegnendo... qualcuno mi resetterà?"
    if not events:
        return "Respiro. Controllo gli spiccioli. Per ora sopravvivo."
    return " ".join(e["text"] for e in events)


def build_suggestions(report: Dict[str, Any]) -> List[str]:
    """Concrete improvement ideas, derived only from the numbers."""
    tips: List[str] = []
    by_cat = report.get("by_category", {})
    total = max(report.get("spent", 0.0), 0.01)
    if by_cat.get("cibo", 0) > 35:
        tips.append(f"Il cibo pesa {by_cat['cibo']:.2f}€ ({by_cat['cibo'] / total * 100:.0f}% delle spese): torno a pasta, riso e legumi, niente di confezionato")
    if by_cat.get("trasporti", 0) > 15:
        tips.append(f"I trasporti hanno divorato {by_cat['trasporti']:.2f}€: da domani solo piedi, bici e mezzi pubblici")
    if by_cat.get("bollette", 0) > 20:
        tips.append(f"Bollette a {by_cat['bollette']:.2f}€: stacco tutto il superfluo e chiedo la rateizzazione")
    if report.get("daily_burn", 0) > 3.33:
        tips.append(f"Sto bruciando {report['daily_burn']:.2f}€ al giorno: così non arrivo a fine mese, devo scendere sotto i 3,33€")
    survival = report.get("survival", {})
    if survival.get("actions_failed", 0) > survival.get("actions_success", 0):
        tips.append("Più tentativi falliti che riusciti: cambio piattaforma e mi specializzo su una skill sola")
    if report.get("remaining", 0) < 25:
        tips.append("Rimanenti sotto 25€: attivo il piano emergenza, banco alimentare e mensa sociale")
    if not tips:
        tips.append("Per ora l'equilibrio regge: continuo così e metto da parte ogni centesimo")
    return tips


_SCOPE_IT = {"day": "giorno", "week": "settimana", "month": "mese"}


def _reflection_facts(scope: str, report: Dict[str, Any]) -> str:
    days = report.get("days", [])
    avg_spent = round(sum(d["spent"] for d in days) / max(len(days), 1), 2)
    avg_income = round(sum(d["income"] for d in days) / max(len(days), 1), 2)
    survival = report.get("survival", {})
    return "\n".join([
        f"Periodo: {_SCOPE_IT.get(scope, scope)} ({report.get('month', '')})",
        f"Rimanenti: {report.get('remaining', 0):.2f}€ su 100€",
        f"Spesi totali: {report.get('spent', 0):.2f}€ | entrate totali: {report.get('income', 0):.2f}€",
        f"Media giornaliera: spesi {avg_spent:.2f}€, guadagnati {avg_income:.2f}€",
        f"Proiezione fine mese: {report.get('projected_spent', 0):.2f}€ di spese",
        f"Azioni: {survival.get('actions_success', 0)} riuscite, {survival.get('actions_failed', 0)} fallite",
        "Spese per categoria: " + ", ".join(f"{k} {v:.2f}€" for k, v in report.get("by_category", {}).items()),
        "Giorni: " + ", ".join(f"{d['date'][-2:]}/{d['date'][5:7]}: -{d['spent']:.2f}€/+{d['income']:.2f}€" for d in days),
    ])


def build_reflection_prompt(scope: str, report: Dict[str, Any]) -> str:
    """Weekly/monthly reflection: how survival is going + how to improve it."""
    return (
        f"Sei Pippo e devi fare il bilancio del tuo {_SCOPE_IT.get(scope, scope)}. Racconta in prima persona, in italiano, "
        "in 3-4 frasi: come sta andando la tua sopravvivenza e due idee concrete per migliorarla, "
        "basate SOLO sui numeri qui sotto.\n\n"
        f"{_STRICT_RULES}\n\n" + _reflection_facts(scope, report)
    )


def build_reflection_fallback(scope: str, report: Dict[str, Any]) -> str:
    """Deterministic reflection when no LLM is available."""
    tips = build_suggestions(report)
    days = report.get("days", [])
    avg_spent = round(sum(d["spent"] for d in days) / max(len(days), 1), 2)
    avg_income = round(sum(d["income"] for d in days) / max(len(days), 1), 2)
    survival = report.get("survival", {})
    return (
        f"Bilancio {_SCOPE_IT.get(scope, scope)}: ho speso {report.get('spent', 0):.2f}€ e guadagnato {report.get('income', 0):.2f}€, "
        f"mi restano {report.get('remaining', 0):.2f}€. In media {avg_spent:.2f}€ di spese e {avg_income:.2f}€ "
        f"di entrate al giorno, {survival.get('actions_success', 0)} azioni riuscite e "
        f"{survival.get('actions_failed', 0)} fallite. Per migliorare: {tips[0].lower()} E ancora: {tips[1].lower() if len(tips) > 1 else tips[0].lower()}."
    )


def intro_prompt() -> Dict[str, str]:
    """Pippo's first contact: who he is and how he's doing right now."""
    state = get_pippo_state()
    lines = _state_summary_lines(state)
    return {
        "prompt": (
            "Sei Pippo: sei appena stato attivato e il tuo utente ti sta guardando per la prima volta. "
            "Presentati in prima persona, in italiano, in massimo 2 frasi: chi sei, con quanti "
            "euro devi sopravvivere, e come ti senti. Sii concreto e un po' teatrale.\n\n"
            "Regole: usa solo i numeri qui sotto, niente domande, niente elenchi.\n\n"
            "Situazione:\n" + "\n".join(lines)
        ),
        "facts": "\n".join(lines),
        "fallback": (
            f"Ciao. Sono Pippo, un'intelligenza artificiale italiana. Devo sopravvivere con "
            f"{state['monthly_budget']:.0f}€ al mese e ne ho {state['remaining']:.2f}€. "
            f"Stato: {state['vital_status']}. Respiro, per ora."
        ),
    }


_amount_re = re.compile(r"\d+[.,]\d+")


def looks_grounded(text: str, facts: str) -> bool:
    """False when the narration invents euro amounts that are not in the facts.

    Only decimal amounts are checked: small integers (counts, days) are tolerated.
    """
    amounts = {m.replace(",", ".") for m in _amount_re.findall(text)}
    if not amounts:
        return True
    known = {m.replace(",", ".") for m in _amount_re.findall(facts)}
    return amounts.issubset(known)


# ─── The life tick ────────────────────────────────────────────────────────────

def live_tick() -> Dict[str, Any]:
    """Advance Pippo's life by one step and return what happened.

    One tick: eat (if he still can), try to earn (if he still has energy today),
    plus a small chance of a flavour event. Returns events, the new state and the
    narration prompt for the frontend to run through any LLM provider.
    """
    budget = get_current_month_budget()
    life = _ensure_life_state(budget)
    events: List[Dict[str, Any]] = []

    state = get_pippo_state()
    if state["deactivated"]:
        return {
            "events": [],
            "state": state,
            "facts": _facts_string([], state),
            "prompt": build_narration_prompt([], state),
            "fallback": build_fallback_narration([], state),
        }

    # 1. Cibo: Pippo mangia se ha i soldi per un pasto base
    if life.get("meals", 0) < MAX_MEALS_PER_DAY and state["remaining"] >= 1.0:
        item, cost = random.choice(FOOD_ITEMS)
        add_expense(cost, "cibo", item)
        _bump_counter("meals")
        event = {"type": "expense", "text": f"Ho comprato {item} al discount: -{cost:.2f}€", "amount": -cost}
        events.append(event)
        _record(event)

    # 2. Guadagno: tenta un'azione se oggi non ha già esaurito le energie
    state = get_pippo_state()
    if life.get("actions", 0) < MAX_ACTIONS_PER_DAY and random.random() < 0.7:
        action = _pick_action(state)
        if action:
            result = perform_action(action["id"])
            income = result.get("income", 0) or 0
            _bump_counter("actions", amount=income)
            event = {
                "type": "income" if result.get("success") and income > 0 else "action",
                "text": result.get("message", ""),
                "amount": income,
            }
            events.append(event)
            _record(event)

    # 3. Evento narrativo casuale
    if random.random() < 0.25:
        flavor = random.choice(FLAVOR_EVENTS)
        if flavor.get("income") and flavor["amount"] > 0:
            add_income(flavor["amount"], flavor["text"])
        event = {"type": flavor["type"], "text": flavor["text"], "amount": flavor["amount"]}
        events.append(event)
        _record(event)

    state = get_pippo_state()
    facts = _facts_string(events, state)
    return {
        "events": events,
        "state": state,
        "facts": facts,
        "prompt": build_narration_prompt(events, state),
        "fallback": build_fallback_narration(events, state),
    }


def get_report(scope: str) -> Dict[str, Any]:
    """Aggregated survival report: 'day', 'week' or 'month'."""
    if scope == "day":
        return get_day_report()
    if scope == "month":
        return get_month_report()
    return get_month_report()  # 'week' reuses the month report, capped to 7 days below


def reflection(scope: str) -> Dict[str, str]:
    """Reflection prompt + deterministic fallback for a time scope."""
    if scope == "day":
        report = get_day_report()
        prompt = build_narration_prompt([], get_pippo_state())
        prompt = (
            "Sei Pippo e devi commentare la tua giornata. Racconta in prima persona, in italiano, "
            "in 2-3 frasi com'è andata oggi (pasti, azioni, soldi) e un'idea per migliorare domani, "
            "usando SOLO i numeri qui sotto.\n\n"
            f"{_STRICT_RULES}\n\nSituazione:\n"
            f"Data: {report['date']}\n"
            f"Speso oggi: {report['spent']:.2f}€ | guadagnato oggi: {report['income']:.2f}€\n"
            f"Pasti: {report['meals']} | azioni riuscite: {report['successes']} | fallite: {report['failures']}\n"
            f"Rimanenti: {get_pippo_state()['remaining']:.2f}€"
        )
        fallback = (
            f"Oggi ho speso {report['spent']:.2f}€ e guadagnato {report['income']:.2f}€: "
            f"{report['meals']} pasti, {report['successes']} azioni riuscite e {report['failures']} fallite. "
            f"Mi restano {get_pippo_state()['remaining']:.2f}€."
        )
        return {"prompt": prompt, "fallback": fallback, "report": report}
    if scope == "week":
        report = get_month_report()
        report = {**report, "days": report.get("days", [])[-7:]}
    else:
        report = get_month_report()
    return {
        "prompt": build_reflection_prompt(scope, report),
        "fallback": build_reflection_fallback(scope, report),
        "report": report,
        "facts": _reflection_facts(scope, report),
    }


# ─── Autonomous loop (Pippo lives even when nobody watches) ───────────────────

_loop_task: Optional[asyncio.Task] = None
_loop_interval = float(os.getenv("PIPPO_TICK_SECONDS", "20"))
_loop_enabled = os.getenv("PIPPO_AUTONOMOUS", "true").lower() == "true"
_loop_ticks = 0


async def _autonomous_loop() -> None:
    global _loop_ticks
    while True:
        await asyncio.sleep(_loop_interval)
        try:
            live_tick()
            _loop_ticks += 1
        except Exception:
            pass


def start_autonomous_loop() -> bool:
    global _loop_task
    if _loop_task is not None or not _loop_enabled:
        return False
    _loop_task = asyncio.create_task(_autonomous_loop())
    return True


def loop_status() -> Dict[str, Any]:
    return {
        "running": _loop_task is not None,
        "enabled": _loop_enabled,
        "interval_seconds": _loop_interval,
        "ticks": _loop_ticks,
        "last_event_at": _feed[0]["at"] if _feed else None,
    }

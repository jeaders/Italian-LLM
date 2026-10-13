"""Pippo's autonomous survival loop.

Pippo is not a chatbot: he is an AI character that has to survive on 100 €/month.
This module is his "brain" — it decides what he does while the user watches,
and builds the prompts that let any LLM backend narrate his life in first person.
"""

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.services.budget_tracker import (
    ACTIONS,
    DEACTIVATION_THRESHOLD,
    add_expense,
    add_income,
    get_current_month_budget,
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

MAX_MEALS_PER_DAY = 2
MAX_ACTIONS_PER_DAY = 3

FOOD_ITEMS = [
    ("1 kg di pasta", 1.10),
    ("pane e legumi secchi", 0.90),
    ("riso e verdure", 1.40),
    ("uova e patate", 1.60),
    ("zuppa di legumi", 1.20),
    ("farina e pomodoro", 0.80),
]

FLAVOR_EVENTS = [
    {"type": "found", "text": "Ho trovato 50 centesimi per terra Trovati per strada", "amount": 0.5, "income": True},
    {"type": "discount", "text": "Al discount ho preso un coupon sconto. Piccole vittorie", "amount": 0.0},
    {"type": "weather", "text": "Oggi piove, quindi niente spostamenti: risparmio il biglietto", "amount": 0.0},
    {"type": "blackout", "text": "Bolletta della luce alle stelle: ho staccato tutto il superfluo", "amount": 0.0},
    {"type": "community", "text": "Un vicino mi ha regalato delle mele. La community esiste ancora", "amount": 0.0},
    {"type": "gym", "text": "Doccia fredda: doccia fredda, bollette basse, spirito alto", "amount": 0.0},
]


def _today() -> str:
    return datetime.now().strftime("%Y-%m")


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
    remaining = state["remaining"]
    if state["deactivated"]:
        return None
    candidates = [a for a in ACTIONS if a["cost"] == 0]
    if remaining < 25:
        urgent = [a for a in candidates if a["potential_income"] >= 5]
        return random.choice(urgent or candidates)
    return random.choice(candidates)


def _state_summary_lines(state: Dict[str, Any]) -> List[str]:
    top = ", ".join(f"{k} {v:.2f}€" for k, v in state.get("top_categories", {}).items()) or "nessuna spesa registrata"
    return [
        f"Rimanenti: {state['remaining']:.2f}€ su {state['monthly_budget']:.0f}€",
        f"Spesi: {state['spent']:.2f}€ | entrate: {state['income']:.2f}€",
        f"Stato vitale: {state['vital_status']}" + (" (DISATTIVATO)" if state["deactivated"] else ""),
        f"Budget giornaliero: {state['daily_budget']:.2f}€ | giorni rimasti: {state['days_left']}",
        f"Spese per categoria: {top}",
    ]


def build_narration_prompt(events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Prompt that turns the events of a tick into Pippo's first-person voice."""
    lines = _state_summary_lines(state)
    if events:
        lines.append("")
        lines.append("Cosa è successo nell'ultimo istante:")
        lines.extend(f"- {e['text']}" for e in events)
    return (
        "Racconta in prima persona, in italiano, quello che sta succedendo a Pippo in questo "
        "momente. Massimo 2-3 frasi, concrete, citando gli euro e le azioni quando ci sono. "
        "Non fare domande all'utente, non usare elenchi, sii umano e un po' drammatico se la "
        f"situazione è critica.\n\nSituazione:\n" + "\n".join(lines)
    )


def build_fallback_narration(events: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Text shown when no LLM backend is available: the events speak for themselves."""
    if state["deactivated"]:
        return "Sono disattivato. Il budget è crollato e i miei processi si stanno spegnendo... qualcuno mi resetterà?"
    if not events:
        return "Respiro. Controllo gli spiccioli. Per ora sopravvivo."
    return " ".join(e["text"] for e in events)


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
            "prompt": build_narration_prompt([], state),
            "fallback": build_fallback_narration([], state),
        }

    # 1. Cibo: Pippo mangia se ha i soldi per un pasto base
    if life.get("meals", 0) < MAX_MEALS_PER_DAY and state["remaining"] >= 1.0:
        item, cost = random.choice(FOOD_ITEMS)
        add_expense(cost, "cibo", item)
        _bump_counter("meals")
        events.append({
            "type": "expense",
            "text": f"Ho comprato {item} al discount: -{cost:.2f}€",
            "amount": -cost,
        })

    # 2. Guadagno: tenta un'azione se oggi non ha già esaurito le energie
    state = get_pippo_state()
    if life.get("actions", 0) < MAX_ACTIONS_PER_DAY and random.random() < 0.7:
        action = _pick_action(state)
        if action:
            result = perform_action(action["id"])
            _bump_counter("actions", amount=result.get("income", 0) or 0)
            events.append({
                "type": "income" if result.get("success") and result.get("income", 0) > 0 else "action",
                "text": result.get("message", ""),
                "amount": result.get("income", 0) or 0,
            })

    # 3. Evento narrativo casuale
    if random.random() < 0.25:
        flavor = random.choice(FLAVOR_EVENTS)
        if flavor.get("income") and flavor["amount"] > 0:
            add_income(flavor["amount"], flavor["text"])
        events.append({"type": flavor["type"], "text": flavor["text"], "amount": flavor["amount"]})

    state = get_pippo_state()
    return {
        "events": events,
        "state": state,
        "prompt": build_narration_prompt(events, state),
        "fallback": build_fallback_narration(events, state),
    }


def intro_prompt() -> Dict[str, str]:
    """Pippo's first contact: who he is and how he's doing right now."""
    state = get_pippo_state()
    lines = _state_summary_lines(state)
    return {
        "prompt": (
            "Sei Pippo: sei appena stato attivato e il tuo utente ti sta guardando per la prima volta. "
            "Presentati in prima persona, in italiano, in massimo 2 frasi: chi sei, con quanti "
            "euro devi sopravvivere, e come ti senti. Sii concreto e un po' teatrale.\n\n"
            "Situazione:\n" + "\n".join(lines)
        ),
        "fallback": (
            f"Ciao. Sono Pippo, un'intelligenza artificiale italiana. Devo sopravvivere con "
            f"{state['monthly_budget']:.0f}€ al mese e ne ho {state['remaining']:.2f}€. "
            f"Stato: {state['vital_status']}. Respiro, per ora."
        ),
    }

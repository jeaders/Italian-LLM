import re
import math
import httpx
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def tool_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "Nessun risultato trovato."
        return "\n\n".join([f"- [{r['title']}]({r['href']})\n  {r['body']}" for r in results])
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Errore nella ricerca web: {str(e)}"


def tool_calculator(expression: str) -> str:
    allowed = set("0123456789+-*/().,%s ")
    safe_expr = "".join(c for c in expression if c in allowed)
    try:
        result = eval(safe_expr, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Errore nel calcolo: {str(e)}"


def tool_wikipedia(query: str) -> str:
    try:
        url = "https://it.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return "Nessun risultato su Wikipedia."
        return "\n\n".join([f"- {r['title']}: {r['snippet']}" for r in results])
    except Exception as e:
        return f"Errore nella ricerca Wikipedia: {str(e)}"


def tool_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            data = resp.json()
        current = data["current_condition"][0]
        return f"Meteo a {city}: {current['weatherDesc'][0]['value']}, {current['temp_C']}°C, umidità {current['humidity']}%"
    except Exception as e:
        return f"Impossibile ottenere il meteo per {city}: {str(e)}"


def tool_datetime() -> str:
    return datetime.now().strftime("Data e ora corrente: %d/%m/%Y %H:%M:%S")


def tool_unit_converter(args: Dict[str, Any]) -> str:
    value = float(args.get("value", 0))
    from_unit = args.get("from_unit", "").lower()
    to_unit = args.get("to_unit", "").lower()
    conversions = {
        ("km", "miglia"): value * 0.621371,
        ("km", "miles"): value * 0.621371,
        ("celsius", "fahrenheit"): value * 9/5 + 32,
        ("euro", "dollaro"): value * 1.08,
        ("kg", "lbs"): value * 2.20462,
        ("litri", "galloni"): value * 0.264172,
    }
    key = (from_unit, to_unit)
    if key in conversions:
        return f"{value} {from_unit} = {conversions[key]:.2f} {to_unit}"
    return f"Conversione da {from_unit} a {to_unit} non supportata."


def detect_tool_call(message: str) -> tuple[str, Dict[str, Any]]:
    msg = message.lower()
    if any(word in msg for word in ["calcola", "quanto fa", "risolvi"]):
        expr = re.sub(r"[^\d+\-*/().%s ]", "", message)
        return "calculator", {"expression": expr or message}
    if any(word in msg for word in ["che ore sono", "data", "ora"]):
        return "datetime", {}
    if any(word in msg for word in ["meteo", "tempo", "weather"]):
        city_match = re.search(r"a\s+([A-Za-zÀ-ÿ\s]+)", message)
        city = city_match.group(1).strip() if city_match else "Roma"
        return "weather", {"city": city}
    if any(word in msg for word in ["converti", "convert"]):
        return "unit_converter", {"value": 1, "from_unit": "km", "to_unit": "miglia"}
    return "web_search", {"query": message}

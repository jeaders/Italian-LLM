import ast
import operator
import re
import httpx
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ITALIAN_OPS = [
    ("elevato alla", "**"),
    ("alla", "**"),
    ("per", "*"),
    ("diviso", "/"),
    ("divisa", "/"),
    ("più", "+"),
    ("piu", "+"),
    ("meno", "-"),
    ("x", "*"),
]


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


def _normalize_expression(expression: str) -> str:
    """Translate common Italian arithmetic words into operators."""
    text = expression.lower()
    for word, op in _ITALIAN_OPS:
        text = re.sub(rf"\b{re.escape(word)}\b", op, text)
    text = text.replace(",", ".")
    return "".join(c for c in text if c in "0123456789+-*/(). %s")


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("costante non numerica")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"operazione non supportata: {type(node).__name__}")


def tool_calculator(expression: str) -> str:
    safe_expr = _normalize_expression(expression)
    if not safe_expr.strip():
        return "Errore nel calcolo: nessuna espressione riconosciuta"
    try:
        tree = ast.parse(safe_expr, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        return f"Errore nel calcolo: {e}"


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

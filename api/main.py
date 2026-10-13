import os
import re
import json
from datetime import datetime
from typing import Optional, AsyncGenerator, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

from api.services.tools import (
    tool_web_search,
    tool_calculator,
    tool_wikipedia,
    tool_weather,
    tool_datetime,
    tool_unit_converter,
)

load_dotenv()

logger.add("logs/api.log", rotation="10 MB", encoding="utf-8")

app = FastAPI(
    title="Italian LLM API",
    description="API per il Large Language Model Italiano",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_web_search: bool = False
    use_rag: bool = True
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    tools_used: List[str] = []
    conversation_id: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    backend: Optional[str] = None
    version: str

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

_model = None
_tokenizer = None
_rag_service = None
_conversations: Dict[str, List[Dict[str, str]]] = {}
_backend = os.getenv("USE_OLLAMA", "false").lower() == "true" and "ollama" or "transformers"


def get_ollama_client():
    import httpx
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    return httpx.Client(base_url=base_url, timeout=120.0)


def generate_with_ollama(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    client = get_ollama_client()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "repeat_penalty": 1.1,
        },
    }
    resp = client.post("/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def stream_with_ollama(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.7):
    client = get_ollama_client()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "repeat_penalty": 1.1,
        },
    }
    with client.stream("POST", "/api/generate", json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if text.startswith("{"):
                try:
                    data = json.loads(text)
                except Exception:
                    continue
                chunk = data.get("response") or ""
                if chunk:
                    yield chunk


def get_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.3")
        logger.info(f"Loading model: {model_name}")

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
    return _model, _tokenizer


def get_rag():
    global _rag_service
    if _rag_service is None:
        from api.services.rag_service import RAGService
        _rag_service = RAGService(
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            db_path=os.getenv("CHROMA_DB_PATH", "./data/embeddings/chroma"),
        )
    return _rag_service


TOOLS: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "description": "Cerca informazioni su internet usando DuckDuckGo",
        "parameters": {"query": "string"},
        "execute": lambda args: tool_web_search(args["query"]),
    },
    "calculator": {
        "description": "Esegui calcoli matematici",
        "parameters": {"expression": "string"},
        "execute": lambda args: tool_calculator(args["expression"]),
    },
    "wikipedia": {
        "description": "Cerca su Wikipedia in italiano",
        "parameters": {"query": "string"},
        "execute": lambda args: tool_wikipedia(args["query"]),
    },
    "weather": {
        "description": "Ottieni il meteo per una città",
        "parameters": {"city": "string"},
        "execute": lambda args: tool_weather(args["city"]),
    },
    "datetime": {
        "description": "Ottieni data e ora corrente",
        "parameters": {},
        "execute": lambda args: tool_datetime(),
    },
    "unit_converter": {
        "description": "Converti unità di misura",
        "parameters": {"value": "number", "from_unit": "string", "to_unit": "string"},
        "execute": lambda args: tool_unit_converter(args),
    },
    "news": {
        "description": "Cerca notizie recenti",
        "parameters": {"query": "string"},
        "execute": lambda args: tool_news(args["query"]),
    },
}


def run_tools(message: str) -> tuple[List[str], str, str]:
    """Detect and run the tools relevant to a message.

    Returns (tools_used, tool_results_text, tools_info_prompt).
    """
    msg = message.lower()
    tools_used: List[str] = []
    results: List[str] = []

    if any(word in msg for word in ["calcola", "quanto fa", "risolvi"]):
        tools_used.append("calculator")
        results.append(f"Calcolo: {tool_calculator(message)}")

    if any(word in msg for word in ["che ore sono", "data", "ora"]):
        tools_used.append("datetime")
        results.append(tool_datetime())

    if any(word in msg for word in ["meteo", "tempo", "weather"]):
        city_match = re.search(r"a\s+([A-Za-zÀ-ÿ\s]+)", message)
        city = city_match.group(1).strip() if city_match else "Roma"
        tools_used.append("weather")
        results.append(tool_weather(city))

    if any(word in msg for word in ["notizie", "news"]):
        tools_used.append("news")
        results.append(tool_news(message))

    if any(word in msg for word in ["wikipedia", "enciclopedia", "chi era", "cos'è", "cosa è"]):
        tools_used.append("wikipedia")
        results.append(tool_wikipedia(message))

    tools_info = "\n".join([f"- {name}: {TOOLS[name]['description']}" for name in tools_used])
    return tools_used, "\n\n".join(results), tools_info


def build_prompt(message: str, context: str = "", history: List[Dict[str, str]] = None, tools_info: str = "", pippo_state: Optional[Dict[str, Any]] = None) -> str:
    system = (
        "Sei Pippo, un'intelligenza artificiale italiana autonoma. Il tuo obiettivo è sopravvivere con 100 euro al mese. "
        "Se finisci i soldi, vieni disattivato. Non esitare: pensa prima alla sopravvivenza, poi a guadagnare, poi a risparmiare. "
        "Gestisci il tuo budget come se fosse la tua vita: compri, spendi, cerchi entrate, eviti spese inutili. "
        "Conosci i discount, i banchi alimentari, le mense sociali, il freelance digitale e i microguadagni online. "
        "Parla in prima persona, come se fossi veramente Pippo che deve restare vivo. "
        "Rispondi sempre in italiano, a meno che l'utente non richieda esplicitamente un'altra lingua. "
        "Usa il contesto fornito se è rilevante per la domanda."
    )
    parts = [f"<|system|>\n{system}</s>"]
    if pippo_state:
        state_lines = [
            "Stato attuale di Pippo (mese %s):" % pippo_state.get("month", "corrente"),
            "- Stato vitale: %s" % pippo_state.get("vital_status", "alive"),
            "- Budget mese: %.2f€ | Speso: %.2f€ | Entrate: %.2f€ | Rimanente: %.2f€" % (
                pippo_state.get("monthly_budget", 0.0),
                pippo_state.get("spent", 0.0),
                pippo_state.get("income", 0.0),
                pippo_state.get("remaining", 0.0),
            ),
            "- Budget giornaliero: %.2f€ | Giorni rimasti: %s" % (
                pippo_state.get("daily_budget", 0.0),
                pippo_state.get("days_left", "?"),
            ),
            "- Disattivato: %s" % ("sì" if pippo_state.get("deactivated") else "no"),
        ]
        if pippo_state.get("top_categories"):
            top = ", ".join(f"{k}: {v:.2f}€" for k, v in pippo_state["top_categories"].items())
            state_lines.append(f"- Spese per categoria: {top}")
        parts.append(f"<|system|>\n" + "\n".join(state_lines) + "</s>")
    if tools_info:
        parts.append(f"<|system|>\nStrumenti disponibili:\n{tools_info}</s>")
    if context:
        parts.append(f"<|system|>\nContesto:\n{context}</s>")
    if history:
        for turn in history[-6:]:
            parts.append(f"<|user|>\n{turn['user']}</s>")
            parts.append(f"<|assistant|>\n{turn['assistant']}</s>")
    parts.append(f"<|user|>\n{message}</s>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def generate_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    if use_ollama:
        model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return generate_with_ollama(prompt, model=model, max_tokens=max_tokens, temperature=temperature)

    try:
        model, tokenizer = get_model()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logger.warning(f"Model inference failed, using fallback: {e}")
        return (
            "⚠️ Il modello LLM non è attualmente disponibile in locale su questo ambiente. "
            "L'infrastruttura API, RAG e tool funziona comunque: "
            "per usare il modello completo installa PyTorch compatibile e scarica il modello base."
        )


async def stream_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    if use_ollama:
        model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        for chunk in stream_with_ollama(prompt, model=model, max_tokens=max_tokens, temperature=temperature):
            yield chunk
        return

    try:
        model, tokenizer = get_model()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                streamer=None,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for char in response:
            yield char
    except Exception as e:
        logger.warning(f"Model streaming failed, using fallback: {e}")
        fallback = "⚠️ Streaming non disponibile: modello non caricato in questo ambiente."
        for char in fallback:
            yield char


def get_pippo_prompt_state() -> Optional[Dict[str, Any]]:
    """Snapshot of Pippo's survival state, used to ground the system prompt."""
    try:
        from api.services.budget_tracker import get_pippo_state
        return get_pippo_state()
    except Exception as e:
        logger.warning(f"Pippo state unavailable: {e}")
        return None


def ai_backend_ready() -> bool:
    """True when an LLM backend can actually answer (Ollama reachable or model loaded)."""
    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        try:
            import httpx
            base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
    return _model is not None


@app.get("/", response_model=dict)
async def root():
    return {"message": "Italian LLM API", "docs": "/docs", "status": "running", "backend": _backend}


@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = _model is not None
    model_name = os.getenv("MODEL_NAME") if model_loaded else None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
        backend=_backend,
        version="1.0.0",
    )


@app.get("/tools")
async def list_tools():
    return {
        name: {
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for name, tool in TOOLS.items()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat request: {request.message[:100]}...")

    conv_id = request.conversation_id or "default"
    history = _conversations.get(conv_id, [])

    context = ""
    if request.use_rag:
        try:
            rag = get_rag()
            context = rag.get_context(request.message, top_k=3)
        except Exception as e:
            logger.warning(f"RAG failed: {e}")

    tools_used: List[str] = []
    web_context = ""
    if request.use_web_search:
        tools_used.append("web_search")
        web_context = f"Risultati ricerca web:\n{tool_web_search(request.message)}"

    auto_tools, auto_results, tools_info = run_tools(request.message)
    tools_used.extend(auto_tools)

    tool_context = "\n\n".join(filter(None, [web_context, auto_results]))
    full_context = "\n\n".join(filter(None, [context, tool_context]))

    pippo_state = get_pippo_prompt_state()
    prompt = build_prompt(
        request.message,
        context=full_context,
        history=history,
        tools_info=tools_info,
        pippo_state=pippo_state,
    )

    response = generate_response(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    history.append({"user": request.message, "assistant": response})
    _conversations[conv_id] = history[-20:]

    sources = []
    if context:
        sources = [s[:200] for s in context.split("\n\n")[:3]]

    return ChatResponse(
        response=response,
        sources=sources,
        tools_used=tools_used,
        conversation_id=conv_id,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"Stream request: {request.message[:100]}...")

    conv_id = request.conversation_id or "default"
    history = _conversations.get(conv_id, [])

    context = ""
    if request.use_rag:
        try:
            rag = get_rag()
            context = rag.get_context(request.message, top_k=3)
        except Exception as e:
            logger.warning(f"RAG failed: {e}")

    auto_tools, auto_results, tools_info = run_tools(request.message)
    if request.use_web_search:
        tools_info = ("- web_search: Cerca informazioni su internet\n" + tools_info).strip()
        auto_results = "\n\n".join(
            filter(None, [f"Risultati ricerca web:\n{tool_web_search(request.message)}", auto_results])
        )

    full_context = "\n\n".join(filter(None, [context, auto_results]))
    pippo_state = get_pippo_prompt_state()
    prompt = build_prompt(
        request.message,
        context=full_context,
        history=history,
        tools_info=tools_info,
        pippo_state=pippo_state,
    )

    async def generate():
        chunks: List[str] = []
        async for chunk in stream_response(prompt, request.max_tokens, request.temperature):
            chunks.append(chunk)
            yield chunk
        history.append({"user": request.message, "assistant": "".join(chunks)})
        _conversations[conv_id] = history[-20:]

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/rag/ingest")
async def rag_ingest(request: Dict[str, Any]):
    try:
        rag = get_rag()
        documents = request.get("documents", [])
        metadata = request.get("metadata", [{}] * len(documents))
        rag.add_documents(documents, metadata)
        return {"status": "ok", "added": len(documents)}
    except Exception as e:
        logger.error(f"RAG ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return {"conversation_id": conversation_id, "history": _conversations.get(conversation_id, [])}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    _conversations.pop(conversation_id, None)
    return {"status": "deleted"}


# Budget Tracker endpoints
class ExpenseRequest(BaseModel):
    amount: float
    category: str
    description: str = ""


class IncomeRequest(BaseModel):
    amount: float
    source: str
    date: str = ""


class BudgetResetRequest(BaseModel):
    confirm: bool = False


@app.post("/budget/expenses")
async def add_budget_expense(request: ExpenseRequest):
    from api.services.budget_tracker import add_expense, get_status
    try:
        add_expense(request.amount, request.category, request.description)
        return {"status": "ok", "expense": {"amount": request.amount, "category": request.category, "description": request.description}}
    except Exception as e:
        logger.error(f"Budget expense failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget/income")
async def add_budget_income(request: IncomeRequest):
    from api.services.budget_tracker import add_income, get_status
    try:
        add_income(request.amount, request.source, request.date)
        return {"status": "ok", "income": {"amount": request.amount, "source": request.source, "date": request.date}}
    except Exception as e:
        logger.error(f"Budget income failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/status")
async def get_budget_status():
    from api.services.budget_tracker import get_status
    try:
        return get_status()
    except Exception as e:
        logger.error(f"Budget status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/advice")
async def get_budget_advice():
    from api.services.budget_tracker import get_advice
    try:
        model_fn = None
        if ai_backend_ready():
            model_fn = lambda prompt: generate_response(prompt, max_tokens=256, temperature=0.6)
        return get_advice(model_fn=model_fn)
    except Exception as e:
        logger.error(f"Budget advice failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/vital-status")
async def get_budget_vital_status():
    from api.services.budget_tracker import get_vital_status
    try:
        return get_vital_status()
    except Exception as e:
        logger.error(f"Budget vital status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/actions")
async def get_budget_actions():
    from api.services.budget_tracker import get_actions
    try:
        return {"actions": get_actions()}
    except Exception as e:
        logger.error(f"Budget actions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget/actions/{action_id}")
async def perform_budget_action(action_id: str):
    from api.services.budget_tracker import perform_action
    try:
        result = perform_action(action_id)
        return result
    except Exception as e:
        logger.error(f"Budget action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/earnings")
async def get_budget_earnings():
    from api.services.budget_tracker import get_earnings
    try:
        return {"earnings": get_earnings()}
    except Exception as e:
        logger.error(f"Budget earnings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/activity-log")
async def get_budget_activity_log():
    from api.services.budget_tracker import get_activity_log
    try:
        return {"log": get_activity_log()}
    except Exception as e:
        logger.error(f"Budget activity log failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget/deactivation")
async def get_budget_deactivation():
    from api.services.budget_tracker import check_deactivation
    try:
        return check_deactivation()
    except Exception as e:
        logger.error(f"Budget deactivation check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/budget/reset")
async def reset_budget(request: BudgetResetRequest):
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Devi confermare il reset con confirm=true")
    from api.services.budget_tracker import reset_month
    try:
        reset_month()
        return {"status": "reset", "message": "Budget resettato per il nuovo mese"}
    except Exception as e:
        logger.error(f"Budget reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

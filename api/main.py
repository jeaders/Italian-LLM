import os
import re
import json
import math
import httpx
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

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
    version: str

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# ── Globals (lazy-loaded) ─────────────────────────────────────────────────────

_model = None
_tokenizer = None
_rag_service = None
_conversations: Dict[str, List[Dict[str, str]]] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Tool Implementations ─────────────────────────────────────────────────────

TOOLS: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "description": "Cerca informazioni su internet usando DuckDuckGo",
        "parameters": {"query": "string"},
        "execute": lambda args: tool_web_search(args["query"]),
    },
    "calculator": {
        "description": "Esegui calcoli matematici. Supporta +, -, *, /, **, sqrt, log, sin, cos, etc.",
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
        "description": "Converti unità di misura (es: km->miles, celsius->fahrenheit)",
        "parameters": {"value": "number", "from_unit": "string", "to_unit": "string"},
        "execute": lambda args: tool_unit_converter(args),
    },
}


def tool_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "Nessun risultato trovato."
        formatted = []
        for r in results:
            formatted.append(f"- [{r['title']}]({r['href']})\n  {r['body']}")
        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Errore nella ricerca web: {str(e)}"


def tool_calculator(expression: str) -> str:
    allowed = set("0123456789+-*/().%s ")
    safe_expr = "".join(c for c in expression if c in allowed)
    try:
        result = eval(safe_expr, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Errore nel calcolo: {str(e)}"


def tool_wikipedia(query: str) -> str:
    try:
        url = "https://it.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
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
    if from_unit == "km" and to_unit in ("miglia", "miles"):
        return f"{value} km = {value * 0.621371:.2f} miglia"
    if from_unit == "celsius" and to_unit == "fahrenheit":
        return f"{value}°C = {value * 9/5 + 32:.1f}°F"
    if from_unit == "euro" and to_unit == "dollaro":
        return f"€{value} ≈ ${value * 1.08:.2f} (stima)"
    return f"Conversione da {from_unit} a {to_unit} non supportata."


# ── Inference ────────────────────────────────────────────────────────────────

def build_prompt(message: str, context: str = "", history: List[Dict[str, str]] = None, tools_info: str = "") -> str:
    system = (
        "Sei un assistente AI italiano competente e utile. "
        "Rispondi sempre in italiano, a meno che l'utente non richieda esplicitamente un'altra lingua. "
        "Usa il contesto fornito se è rilevante per la domanda."
    )
    parts = [f"<|system|>\n{system}</s>"]
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


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=dict)
async def root():
    return {"message": "Italian LLM API", "docs": "/docs", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = _model is not None
    model_name = os.getenv("MODEL_NAME") if model_loaded else None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
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

    # 1. Retrieve history
    conv_id = request.conversation_id or "default"
    history = _conversations.get(conv_id, [])

    # 2. RAG context
    context = ""
    if request.use_rag:
        try:
            rag = get_rag()
            context = rag.get_context(request.message, top_k=3)
        except Exception as e:
            logger.warning(f"RAG failed: {e}")

    # 3. Tool use (simple rule-based routing)
    tools_used = []
    tool_results = []

    if request.use_web_search:
        tools_used.append("web_search")
        tool_results.append(f"Risultati ricerca web:\n{tool_web_search(request.message)}")

    if any(word in request.message.lower() for word in ["calcola", "quanto fa", "risolvi", "calcola"]):
        expr = re.sub(r"[^\d+\-*/().%s ]", "", request.message)
        if expr:
            tools_used.append("calculator")
            tool_results.append(f"Calcolo: {tool_calculator(expr)}")

    if "che ore sono" in request.message.lower() or "data" in request.message.lower():
        tools_used.append("datetime")
        tool_results.append(tool_datetime())

    if not request.use_web_search and any(word in request.message.lower() for word in ["meteo", "tempo", "weather"]):
        city_match = re.search(r"a\s+([A-Za-zÀ-ÿ\s]+)", request.message)
        city = city_match.group(1) if city_match else "Roma"
        tools_used.append("weather")
        tool_results.append(tool_weather(city))

    tools_info = "\n".join([f"- {name}: {TOOLS[name]['description']}" for name in tools_used])
    tool_context = "\n\n".join(tool_results) if tool_results else ""

    # 4. Build prompt
    full_context = "\n\n".join(filter(None, [context, tool_context]))
    prompt = build_prompt(request.message, context=full_context, history=history, tools_info=tools_info)

    # 5. Generate
    response = generate_response(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    # 6. Update history
    history.append({"user": request.message, "assistant": response})
    _conversations[conv_id] = history[-20:]

    # 7. Sources
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

    tools_info = ""
    tool_results = []
    if request.use_web_search:
        tool_results.append(f"Risultati ricerca web:\n{tool_web_search(request.message)}")
        tools_info += "\n- web_search: Cerca informazioni su internet"

    full_context = "\n\n".join(filter(None, [context, "\n\n".join(tool_results)]))
    prompt = build_prompt(request.message, context=full_context, history=history, tools_info=tools_info)

    async def generate():
        async for chunk in stream_response(prompt, request.max_tokens, request.temperature):
            yield chunk
        # Update history after stream
        # Note: we'd need to accumulate the full response to store it
        # For simplicity, skip history update in streaming mode

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

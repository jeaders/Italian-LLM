import gradio as gr
import os
import requests
import re
import math
from datetime import datetime

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2-2b-it")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def query_model(prompt: str) -> str:
    if not HF_TOKEN:
        return "⚠️ Configura HF_TOKEN nelle variabili d'ambiente dello Space."
    try:
        resp = requests.post(API_URL, headers=HEADERS, json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False},
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        return str(data)
    except Exception as e:
        return f"Errore nella chiamata al modello: {str(e)}"


def tool_calculator(expression: str) -> str:
    allowed = set("0123456789+-*/().,%s ")
    safe_expr = "".join(c for c in expression if c in allowed)
    try:
        result = eval(safe_expr, {"__builtins__": {}}, {"math": math, "sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Errore nel calcolo: {str(e)}"


def tool_datetime() -> str:
    return datetime.now().strftime("Data e ora corrente: %d/%m/%Y %H:%M:%S")


def tool_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        return f"Meteo a {city}: {current['weatherDesc'][0]['value']}, {current['temp_C']}°C, umidità {current['humidity']}%"
    except Exception as e:
        return f"Impossibile ottenere il meteo per {city}: {str(e)}"


def chat(message: str, history, use_web_search: bool, use_rag: bool, temperature: float):
    if not message.strip():
        return history, ""

    tools_used = []
    tool_results = []

    if any(word in message.lower() for word in ["calcola", "quanto fa", "risolvi"]):
        expr = re.sub(r"[^\d+\-*/().%s ]", "", message)
        if expr:
            tools_used.append("calculator")
            tool_results.append(f"Calcolo: {tool_calculator(expr)}")

    if any(word in message.lower() for word in ["che ore sono", "data", "ora"]):
        tools_used.append("datetime")
        tool_results.append(tool_datetime())

    if any(word in message.lower() for word in ["meteo", "tempo", "weather"]):
        city_match = re.search(r"a\s+([A-Za-zÀ-ÿ\s]+)", message)
        city = city_match.group(1).strip() if city_match else "Roma"
        tools_used.append("weather")
        tool_results.append(tool_weather(city))

    system = "Sei un assistente AI italiano competente e utile. Rispondi sempre in italiano."
    prompt = f"<|system|>\n{system}</s>\n"
    if tool_results:
        prompt += f"<|system|>\nStrumenti:\n" + "\n".join(tool_results) + "</s>\n"
    prompt += f"<|user|>\n{message}</s>\n<|assistant|>"

    response = query_model(prompt)

    history = history or []
    history.append((message, response))
    return history, ""


with gr.Blocks(title="Italian LLM") as demo:
    gr.Markdown("# 🇮🇹 Italian LLM\nAssistente AI italiano pubblico con strumenti.")
    chatbot = gr.Chatbot(height=520, label="Chat")
    msg = gr.Textbox(label="Messaggio", placeholder="Es: Spiega la relatività, Calcola 128×56, Che ore sono?, Meteo a Milano")
    with gr.Row():
        use_web_search = gr.Checkbox(label="Ricerca web", value=False)
        use_rag = gr.Checkbox(label="RAG", value=False)
        temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperatura")
    clear = gr.Button("Cancella chat")
    state = gr.State([])

    msg.submit(chat, [msg, state, use_web_search, use_rag, temperature], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch()

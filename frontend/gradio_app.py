import gradio as gr
import httpx
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

DESCRIPTION = """
# 🇮🇹 Italian LLM
Assistente AI italiano con ricerca web, calcoli e RAG.
"""

CSS = """
.gradio-container {font-family: system-ui, -apple-system, sans-serif;}
footer {visibility: hidden;}
"""


async def chat(message: str, history: list, use_web_search: bool, use_rag: bool):
    if not message.strip():
        return history, ""
    payload = {
        "message": message,
        "conversation_id": "space-demo",
        "use_web_search": use_web_search,
        "use_rag": use_rag,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{API_URL}/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    reply = data.get("response", "Errore nella risposta.")
    history = history + [(message, reply)]
    return history, ""


with gr.Blocks(css=CSS, title="Italian LLM") as demo:
    gr.Markdown(DESCRIPTION)
    chatbot = gr.Chatbot(height=500, label="Chat")
    msg = gr.Textbox(label="Scrivi un messaggio", placeholder="Es: Spiega la relatività generale")
    with gr.Row():
        use_web_search = gr.Checkbox(label="Ricerca web", value=False)
        use_rag = gr.Checkbox(label="RAG", value=False)
        clear = gr.Button("Cancella")
    state = gr.State([])

    msg.submit(chat, [msg, state, use_web_search, use_rag], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

import chainlit as cl
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="🇮🇹 Ciao! Sono il tuo assistente AI italiano. Come posso aiutarti?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{API_URL}/chat",
                json={
                    "message": message.content,
                    "use_web_search": False,
                    "use_rag": False,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            await msg.stream_token(data["response"])
            
            if data.get("sources"):
                source_elements = [
                    cl.Text(name=f"Fonte {i+1}", content=source)
                    for i, source in enumerate(data["sources"])
                ]
                await cl.Message(
                    content="📚 Fonti:",
                    elements=source_elements
                ).send()
                
        except Exception as e:
            await msg.stream_token(f"Errore: {str(e)}")

# Italian LLM 🇮🇹

[![HuggingFace Space](https://huggingface.co/api/spaces/jeaders/Italian-LLM-Space/badge)](https://huggingface.co/spaces/jeaders/Italian-LLM-Space)

An open-source, Italian-first AI assistant stack: a FastAPI backend, a streaming chat UI, a RAG
pipeline, tool use, and **Pippo** — an autonomous AI character that has to survive on 100 €/month.

> **Project status: active work in progress.** The API, the survival simulation and the tool layer
> are working and tested. The LLM layer is provider-based (Ollama / Groq / WebLLM / local
> transformers): bring any model you like, Pippo supplies the personality, the memory and the
> survival loop.

## ✨ Features

- **Chat backend (FastAPI)** with `/chat` and `/chat/stream`, conversation history, RAG context and
  automatic tool routing
- **Pippo survival simulation** — a budget tracker played as a life-or-death game: vital status,
  expenses, income, simulated freelance actions, earnings opportunities, activity log and
  deactivation when the budget collapses
- **Grounded personality** — Pippo's real budget state (remaining, daily budget, top categories,
  vital status) is injected into the system prompt, so the model speaks about its actual situation
- **AI advice with a rules fallback** — `/budget/advice` uses the configured LLM when one is
  reachable, otherwise the built-in survival heuristics
- **RAG** with ChromaDB and multilingual embeddings, plus document ingest endpoint
- **Tools**: web search (DuckDuckGo), calculator (safe AST evaluator that understands "128 per 56"),
  Italian Wikipedia, weather, news, date/time, unit conversion
- **Multiple LLM providers**: Ollama (local), transformers (local GPU/CPU), WebLLM (in-browser via
  WebGPU) and Groq (free API key) — selectable from the UI
- **Modern chat UI**: streaming responses, Markdown rendering, conversation search, JSON
  export/import, keyboard shortcuts, dark mode, voice input, mobile-first layout
- **Pippo survival UI**: terminal-style dashboard with vital badge, action cards with simulated
  outcomes, earnings sources and activity log

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Frontend (static, no build step)                         │
│  - frontend/index.html          chat UI (Groq/WebLLM/API) │
│  - frontend/budget_tracker.html Pippo survival dashboard  │
├───────────────────────────────────────────────────────────┤
│  API gateway (FastAPI, api/main.py)                       │
│  - /chat, /chat/stream        grounded prompts + tools    │
│  - /budget/*                  survival simulation         │
│  - /rag/ingest                vector store writes         │
├───────────────────────────────────────────────────────────┤
│  Services                                                  │
│  - api/services/tools.py          web, calc, wiki, meteo… │
│  - api/services/rag_service.py    ChromaDB + embeddings    │
│  - api/services/budget_tracker.py Pippo state machine      │
├───────────────────────────────────────────────────────────┤
│  LLM engines: Ollama · transformers · WebLLM · Groq       │
└───────────────────────────────────────────────────────────┘
```

## 🚀 Quick start

### Prerequisites

- Python 3.11+ (3.12 recommended for training)
- 16 GB RAM minimum
- macOS / Linux / Windows

### Option 1 — with Ollama (recommended for local inference)

```bash
# 1. Install Ollama and pull a model
brew install --cask ollama
ollama pull llama3.1:8b

# 2. Clone and configure
git clone https://github.com/jeaders/Italian-LLM.git
cd Italian-LLM
cp .env.example .env

# 3. In .env
#    USE_OLLAMA=true
#    OLLAMA_URL=http://localhost:11434
#    OLLAMA_MODEL=llama3.1:8b

# 4. Start the API
make inference
# API docs: http://localhost:8000/docs

# 5. Open the UIs
open frontend/index.html
open frontend/budget_tracker.html
```

### Option 2 — full local install

```bash
git clone https://github.com/jeaders/Italian-LLM.git
cd Italian-LLM
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8000
open frontend/index.html
```

Without a local model the API still runs: chat answers with a clear "model unavailable" message,
tools, RAG and the whole Pippo simulation keep working.

## 🧠 Pippo — surviving on 100 €/month

Pippo is the heart of this project: an Italian AI character whose only goal is to stay alive on
100 € per month. The simulation is real state, not decoration:

- **Vital status**: `alive` → `warning` → `danger` → `critical` → `dead`, driven by the remaining
  budget (deactivation below −20 €)
- **Expenses and income** are tracked per calendar month; when the month rolls over the previous
  month is archived to `history` and the budget restarts clean
- **Survival actions** (microtasks, freelance, tutoring, transcription…) have a simulated outcome,
  a cost and an urgency that scales with how close Pippo is to deactivation. A deactivated Pippo
  cannot act: recover income or reset the month
- **Grounded chat**: Pippo's system prompt includes his live state (remaining, spent, income, daily
  budget, days left, top categories), so he complains about *his* budget, not a generic one
- **Advice**: `GET /budget/advice` asks the LLM when one is configured, otherwise uses the built-in
  survival heuristics (`tip_source: "ai" | "rules"` in the response)

State lives in `data/budget/budget_tracker.json` (single-user, file-backed — see the roadmap).

## 📡 API

Base URL: `http://localhost:8000` — interactive docs at `/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check and active backend |
| `GET` | `/tools` | Available tools and their parameters |
| `POST` | `/chat` | Synchronous chat (RAG, tools, Pippo state) |
| `POST` | `/chat/stream` | Streaming chat |
| `GET` | `/conversations/{id}` | Conversation history |
| `DELETE` | `/conversations/{id}` | Delete a conversation |
| `POST` | `/rag/ingest` | Add documents to the vector store |
| `POST` | `/budget/expenses` | Record an expense |
| `POST` | `/budget/income` | Record income |
| `GET` | `/budget/status` | Budget status and alerts |
| `GET` | `/budget/advice` | Survival advice (AI or rules) |
| `GET` | `/budget/vital-status` | Pippo's vital status |
| `GET` | `/budget/actions` | Available survival actions |
| `POST` | `/budget/actions/{action_id}` | Perform a survival action |
| `GET` | `/budget/earnings` | Earning opportunity catalogue |
| `GET` | `/budget/activity-log` | Recent activity log |
| `GET` | `/budget/deactivation` | Deactivation risk and recovery info |
| `POST` | `/budget/reset` | Reset the month (`{"confirm": true}`) |

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quanto fa 128 per 56?", "use_rag": false, "use_web_search": false}'

curl http://localhost:8000/budget/vital-status
curl -X POST http://localhost:8000/budget/actions/microtask
```

## ⚙️ Configuration

Copy `.env.example` to `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `USE_OLLAMA` | `false` | Use Ollama as the LLM backend |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default model |
| `OLLAMA_CHAT_MODEL` | — | Optional separate chat model |
| `MODEL_NAME` | `mistralai/Mistral-7B-v0.3` | transformers model (when not using Ollama) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | RAG embeddings |
| `CHROMA_DB_PATH` | `./data/embeddings/chroma` | Vector store location |

The chat UI also supports Groq and WebLLM directly in the browser (no backend needed) — see
`frontend/index.html`.

## 🧪 Tests

```bash
make test          # or: python -m pytest tests/ -v
```

18 tests covering the API surface, the budget simulation (reset, month rollover, deactivation),
tool routing and the calculator.

## 🚢 Deployment

| Target | Notes |
|--------|-------|
| **Docker** | `docker-compose up --build` serves the API; open the frontend pages from `frontend/` |
| **Fly.io** | `flyctl launch && flyctl deploy` |
| **HuggingFace Spaces** | Gradio/Streamlit entrypoints in `frontend/` and `space/` |
| **Netlify (static)** | Publishes `frontend/` with no build step. Note: the Pippo dashboard calls
  `/budget/*` on its own origin, so it needs the API served from the same origin (or a proxy) —
  chat works standalone via Groq/WebLLM. |

## 📁 Project structure

```
Italian-LLM/
├── api/
│   ├── main.py                  # FastAPI app: chat, tools, budget endpoints
│   └── services/
│       ├── tools.py             # web search, calculator, wikipedia, weather, news…
│       ├── rag_service.py       # ChromaDB + embeddings
│       └── budget_tracker.py    # Pippo survival state machine
├── frontend/
│   ├── index.html               # chat UI (Groq / WebLLM / local API)
│   └── budget_tracker.html      # Pippo survival dashboard
├── training/                    # LoRA/QLoRA pipeline and configs (incl. Pippo SFT)
├── data/                        # datasets and runtime state
├── tests/                       # pytest suite
├── deploy/                      # Docker and deployment files
└── Makefile                     # make inference / train-pippo / test / …
```

## 🗺️ Roadmap

- [ ] SQLite persistence for conversations (today: in-memory, lost on restart)
- [ ] Multi-user budgets (today: single shared budget file)
- [ ] Action cooldowns and daily limits in the survival loop
- [ ] Release a trained Pippo adapter (the survival SFT config already exists)
- [ ] Real token streaming for the local transformers backend

## 🤝 Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co) for the training ecosystem
- [Ollama](https://ollama.ai) for local inference
- [ChromaDB](https://www.trychroma.com) and [sentence-transformers](https://www.sbert.net) for RAG
- [Groq](https://groq.com) and [WebLLM](https://webllm.mlc.ai/) for browser/fast inference options
- The Italian AI community

---

⭐ If you like the project, leave a star on GitHub!

Made with ❤️ in Italy 🇮🇹 by [jeaders](https://github.com/jeaders)

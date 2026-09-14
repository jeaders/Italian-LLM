# Italian LLM 🇮🇹

[![HuggingFace Space](https://huggingface.co/api/spaces/jeaders/Italian-LLM-Space/badge)](https://huggingface.co/spaces/jeaders/Italian-LLM-Space)

Un Large Language Model open source in italiano, con API, RAG, tool use e un assistente specializzato per il budget mensile.

## 🎯 Obiettivo

Creare un LLM italiano accessibile e utile, completamente open source, che possa:
- Comprendere e generare testo italiano naturale
- Eseguire ragionamento complesso
- Accedere a informazioni aggiornate tramite web search
- Eseguire calcoli e rispondere con tool verificati
- Essere deployato gratuitamente su hardware consumer
- Aiutare nella gestione del budget mensile con un assistente dedicato

## ✨ Caratteristiche

- **Modello Base**: fine-tuning instruction-style con LoRA/QLoRA
- **RAG**: Retrieval Augmented Generation per risposte accurate e citate
- **Tool Use**: calcolatrice, web search, Wikipedia, meteo, notizie, conversioni
- **Budget Tracker**: frontend dedicato per tracciare spese e ricevere consigli di sopravvivenza con 100 €/mese
- **UI Moderna**: chat in streaming, cronologia conversazioni, dark mode
- **Deploy Gratuito**: HuggingFace Spaces, Fly.io, Vercel, Docker

## 🏗️ Architettura

```
┌──────────────────────────────────────────────┐
│  Frontend                                   │
│  - Chat UI (frontend/index.html)             │
│  - Budget Tracker (frontend/budget_tracker.html) │
├──────────────────────────────────────────────┤
│  API Gateway (FastAPI)                       │
│  - /chat, /chat/stream                       │
│  - /budget/status, /budget/expenses          │
│  - /budget/advice, /budget/reset             │
├──────────────────────────────────────────────┤
│  LLM Engine                                 │
│  - transformers / Ollama backend             │
│  - LoRA adapter locale                       │
├──────────────────────────────────────────────┤
│  RAG Layer (ChromaDB + Embeddings)           │
├──────────────────────────────────────────────┤
│  Tools (web search, calc, wiki, meteo, news) │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisiti

- Python **3.11 o 3.12** per il training
- Python **3.10+** per eseguire l'API
- 16GB RAM (minimo), 32GB+ consigliato per il training
- macOS / Linux / Windows

### Opzione 1: API + UI con Ollama (consigliata)

```bash
# 1. Installa Ollama
brew install --cask ollama

# 2. Scarica un modello italiano
ollama pull llama3.1:8b

# 3. Clona il repo
git clone https://github.com/jeaders/Italian-LLM.git
cd Italian-LLM
cp .env.example .env

# 4. In .env imposta
USE_OLLAMA=true
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# 5. Avvia API
make inference
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# 6. Apri la UI
open frontend/index.html
# Budget Tracker
open frontend/budget_tracker.html
```

### Opzione 2: Installazione completa

```bash
git clone https://github.com/jeaders/Italian-LLM.git
cd Italian-LLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8000
open frontend/index.html
```

## 🧠 Training: Pippo — sopravvivenza con 100 €/mese

Questo repo include anche un flusso dedicato per addestrare un assistente specializzato nel sopravvivere con 100 € al mese.

### Dataset

- `data/processed/survival_instructions.json`: istruzioni italiane su cibo, affitto, bollette, trasporti, aiuti sociali, baratto e risparmio estremo

### Configurazione consigliata per budget limitato

```yaml
# training/configs/sft_config.yaml
model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_path: "data/processed/survival_instructions.json"
output_dir: "./models/adapter/pippo-survival"
max_seq_length: 1024

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 2e-4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  fp16: true
  optim: "paged_adamw_8bit"

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### Avvia training

```bash
# Setup automatico con Python 3.12
make train-pippo
```

Lo script:
1. Verifica la presenza di Python 3.12
2. Crea/riusa `.venv312`
3. Installa le dipendenze
4. Avvia `training/scripts/train_sft.py`

Se Python 3.12 non è installato:
```bash
brew install python@3.12
make train-pippo
```

## 📊 Dataset

Utilizziamo una combinazione di dataset italiani:

| Dataset | Contenuto | Dimensione |
|---------|-----------|------------|
| Wikipedia Italiano | Enciclopedia | ~1.5M articoli |
| OSCAR-it | Web corpus | ~50GB |
| CulturaX-it | Dati web | ~30GB |
| Testi legali | Normativa italiana | ~5GB |
| Istruzioni custom | QA, reasoning, sopravvivenza | ~100k esempi |

## 🔧 API

### Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| POST | `/chat` | Chat sincrona |
| POST | `/chat/stream` | Chat con streaming |
| GET | `/health` | Health check |
| GET | `/tools` | Lista tool disponibili |
| POST | `/rag/ingest` | Aggiungi documenti al RAG |
| POST | `/budget/expenses` | Aggiungi spesa |
| GET | `/budget/status` | Stato budget mensile |
| GET | `/budget/advice` | Consigli di sopravvivenza |
| POST | `/budget/reset` | Reset budget mensile |

### Esempio

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Spiega la relatività generale",
    "use_rag": true,
    "use_web_search": false,
    "max_tokens": 512
  }'
```

Budget tracker:

```bash
curl -X POST "http://localhost:8000/budget/expenses" \
  -H "Content-Type: application/json" \
  -d '{"amount": 5.5, "category": "cibo", "description": "pasta"}'

curl "http://localhost:8000/budget/status"
curl "http://localhost:8000/budget/advice"
```

## 📦 Deploy pubblico

### HuggingFace Space

Questo repo è predisposto per uno Space pubblico.

1. Vai su [huggingface.co/spaces](https://huggingface.co/spaces)
2. Crea un nuovo Space come **Gradio** o **Streamlit**
3. Collega il repo GitHub `jeaders/Italian-LLM`
4. Configura le variabili d'ambiente
5. Il deploy parte automaticamente

### Locale con Ollama

```bash
brew install --cask ollama
ollama pull llama3.1:8b
cp .env.example .env
uvicorn api.main:app --reload --port 8000
open frontend/index.html
```

### Docker

```bash
docker-compose up --build
```

### Fly.io

```bash
flyctl launch
flyctl deploy
```

## 🧪 Test

```bash
# Avvia l'API in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Esegui i test
python -m pytest tests/ -v

# Oppure con Make
make test
```

## 📁 Struttura progetto

```
Italian-LLM/
├── api/                        # Backend FastAPI
│   ├── main.py                 # Endpoints: chat, budget, health
│   └── services/               # Tool, RAG, budget tracker
├── frontend/                   # UI
│   ├── index.html              # Chat UI
│   └── budget_tracker.html     # Budget Tracker
├── training/                   # Pipeline di training
│   ├── scripts/
│   │   ├── train_sft.py        # Training LoRA/QLoRA
│   │   ├── evaluate.py         # Valutazione modelli
│   │   └── preprocess_data.py  # Preparazione dataset
│   └── configs/
│       └── sft_config.yaml     # Configurazione training
├── data/
│   ├── raw/                    # Dati grezzi
│   └── processed/
│       ├── italian_instructions.json
│       └── survival_instructions.json
├── models/                     # Modelli e adapter
│   ├── base/
│   ├── adapter/
│   └── merged/
├── rag/                        # Retrieval Augmented Generation
├── space/                      # HuggingFace Space
├── tests/
│   └── test_api.py             # Test API + budget
├── notebooks/
│   └── train.ipynb             # Training interattivo
├── deploy/                     # Docker e deploy
├── logs/                       # Log applicazione
├── setup_training.sh           # Setup ambiente training
├── Makefile                    # Comandi rapidi
├── requirements.txt            # Dipendenze
├── docker-compose.yml          # Orchestrazione servizi
├── .env.example                # Configurazione ambiente
└── README.md                   # Questo file
```

## ⚙️ Configurazione

Copia `.env.example` in `.env` e configura:

```bash
# Modello
MODEL_NAME=mistralai/Mistral-7B-v0.3
ADAPTER_PATH=./models/adapter
BASE_MODEL_PATH=./models/base

# Ollama (consigliato per hardware consumer)
USE_OLLAMA=true
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# API
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development

# Embedding
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_DB_PATH=./data/embeddings/chroma

# Web search
DUCKDUCKGO_API_KEY=

# Database
DATABASE_URL=sqlite:///./data/chat_history.db

# Logging
LOG_LEVEL=INFO
```

## 🧰 Comandi Make

```bash
make help           # Mostra tutti i comandi disponibili
make install        # Installa dipendenze
make train          # Addestra modello generico
make train-pippo    # Addestra Pippo su sopravvivenza 100€/mese
make inference      # Avvia API con uvicorn
make ui             # Apri chat UI
make setup-ollama   # Installa Ollama e scarica modello
make deploy         # Deploy con Docker
make test           # Esegui test pytest
make clean          # Pulisci cache e file temporanei
make data           # Scarica dataset
```

## 🤝 Contribuire

Vedi [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 Licenza

MIT License - vedi [LICENSE](LICENSE)

## 🙏 Riconoscimenti

- [Mistral AI](https://mistral.ai) per il modello base
- [Hugging Face](https://huggingface.co) per l'ecosistema
- [LangChain](https://langchain.com) per l'orchestrazione
- [Ollama](https://ollama.ai) per il runtime locale
- La community italiana AI per il supporto

## 📞 Contatti

- GitHub Issues: per bug e feature requests
- Discussions: per domande generali

---

⭐ Se ti piace il progetto, lascia una stella su GitHub!

Made with ❤️ in Italy 🇮🇹 Alex Mirici Web Developer

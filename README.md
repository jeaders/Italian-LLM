# Italian LLM 🇮🇹

Un Large Language Model open source in italiano, addestrato per competere con i migliori modelli internazionali come Claude, GPT-4, Gemini e Perplexity.

## 🎯 Obiettivo

Creare un LLM italiano di alta qualità, completamente open source e gratuito, che possa:
- Comprendere e generare testo italiano naturale
- Eseguire ragionamento complesso
- Accedere a informazioni aggiornate tramite web search
- Eseguire calcoli e tool use
- Essere deployato gratuitamente su hardware consumer

## ✨ Caratteristiche

- **Modello Base**: Fine-tuning di Mistral 7B / Llama 3.1 8B su corpus italiano
- **RAG**: Retrieval Augmented Generation per risposte accurate e citate
- **Tool Use**: Calcolatrice, web search, Wikipedia, meteo, conversioni unità
- **UI Moderna**: Chat con streaming, cronologia conversazioni, dark mode
- **Deploy Gratuito**: HuggingFace Spaces, Fly.io, Vercel

## 🏗️ Architettura

```
┌──────────────────────────────────────────────┐
│  Frontend (Chainlit / Next.js)                │
├──────────────────────────────────────────────┤
│  API Gateway (FastAPI)                        │
├──────────────────────────────────────────────┤
│  LLM Engine (Mistral 7B + LoRA)              │
├──────────────────────────────────────────────┤
│  RAG Layer (ChromaDB + Embeddings)            │
├──────────────────────────────────────────────┤
│  Tools (Web Search, Calculator, Wikipedia)    │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisiti
- Python 3.10+
- 16GB RAM (minimo), 32GB+ consigliato
- GPU NVIDIA con 12GB+ VRAM (opzionale ma consigliato)

### Installazione Locale

```bash
# Clone il repo
git clone https://github.com/tuousername/italian-llm.git
cd italian-llm

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Configura .env
cp .env.example .env
# Modifica .env con le tue configurazioni

# Avvia API
uvicorn api.main:app --reload --port 8000

# Avvia UI (in un altro terminale)
chainlit run frontend/chainlit_app.py --port 8501
```

Apri http://localhost:8501 nel browser.

## 📊 Dataset

Utilizziamo una combinazione di dataset italiani:

| Dataset | Contenuto | Dimensione |
|---------|-----------|------------|
| Wikipedia Italiano | Enciclopedia | ~1.5M articoli |
| OSCAR-it | Web corpus | ~50GB |
| CulturaX-it | Dati web | ~30GB |
| Testi legali | Normativa italiana | ~5GB |
| Istruzioni custom | QA, reasoning | ~100k esempi |

## 🎓 Addestramento

### Setup GPU (Google Colab)

1. Apri [Colab](https://colab.research.google.com)
2. Carica il notebook da `notebooks/train.ipynb`
3. Assicurati di usare GPU T4 o superiore
4. Esegui le celle in ordine

### Training Locale

```bash
# Prepara i dati
python training/scripts/preprocess_data.py

# Avvia training
python training/scripts/train_sft.py --config training/configs/sft_config.yaml
```

### Configurazione Consigliata

```yaml
# training/configs/sft_config.yaml
model_name: "mistralai/Mistral-7B-v0.3"
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 2048
```

## 🔧 API

### Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| POST | `/chat` | Chat sincrona |
| POST | `/chat/stream` | Chat con streaming |
| GET | `/health` | Health check |
| GET | `/tools` | Lista tool disponibili |
| POST | `/rag/ingest` | Aggiungi documenti al RAG |

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

## 📦 Deploy

### HuggingFace Spaces (Consigliato)

1. Crea uno Space su [huggingface.co/spaces](https://huggingface.co/spaces)
2. Carica il modello su [huggingface.co/models](https://huggingface.co/models)
3. Configura lo Space con Gradio o Streamlit
4. Deploy automatico!

### Docker

```bash
docker-compose up --build
```

### Fly.io

```bash
flyctl launch
flyctl deploy
```

## 🧪 Valutazione

```bash
# Esegui evaluation completa
python training/scripts/evaluate.py --model ./models/merged/italian-llm-7b --test data/processed/test_set.json
```

Metriche:
- **Perplexity**: misura della fluency
- **BLEU/ROUGE**: somiglianza con reference
- **BERTScore**: similarità semantica
- **Toxicity**: sicurezza contenuti
- **Latency**: tempo di risposta

## 🤝 Contribuire

Vedi [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 Licenza

MIT License - vedi [LICENSE](LICENSE)

## 🙏 Riconoscimenti

- [Mistral AI](https://mistral.ai) per il modello base
- [Hugging Face](https://huggingface.co) per l'ecosistema
- [LangChain](https://langchain.com) per l'orchestrazione
- La community italiana AI per il supporto

## 📞 Contatti

- GitHub Issues: per bug e feature requests
- Discussions: per domande generali

---

⭐ Se ti piace il progetto, lascia una stella su GitHub!

Made with ❤️ in Italy 🇮🇹

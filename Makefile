.PHONY: help install train inference deploy clean ui setup-ollama deploy-netlify

# Su macOS 'python' spesso non esiste: usa il venv del progetto se c'è, altrimenti python3
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

help:
	@echo "Italian LLM - Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make train         - Start training"
	@echo "  make train-pippo   - Train Pippo on survival dataset (100€ budget)"
	@echo "  make survival-data - Expand survival training dataset"
	@echo "  make inference     - Start API + frontend (open http://localhost:8000/app)"
	@echo "  make ui            - Show how to open the UI"
	@echo "  make setup-ollama  - Install and pull Ollama model"
	@echo "  make deploy        - Deploy with Docker"
	@echo "  make deploy-netlify - Preview/deploy frontend on Netlify"
	@echo "  make clean         - Clean temporary files"
	@echo "  make data          - Download datasets"

install:
	pip install -r requirements.txt

train:
	$(PYTHON) training/scripts/train_sft.py

train-pippo:
	@bash setup_training.sh

survival-data:
	$(PYTHON) scripts/expand_survival_data.py

inference:
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

ui:
	@echo "Con 'make inference' attivo, apri: http://localhost:8000/app"
	@echo "Dashboard sopravvivenza:            http://localhost:8000/app/budget_tracker.html"

setup-ollama:
	@echo "Installing Ollama..."
	brew install --cask ollama
	@echo "Pulling llama3.1:8b model..."
	ollama pull llama3.1:8b
	@echo "Done. Now set USE_OLLAMA=true in .env and run 'make inference'"

deploy:
	docker-compose up --build

deploy-netlify:
	netlify deploy --prod --dir=frontend

data:
	$(PYTHON) data/download_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf logs/*.log
	rm -rf .pytest_cache

test:
	$(PYTHON) -m pytest tests/ -v

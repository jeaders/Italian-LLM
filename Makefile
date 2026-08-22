.PHONY: help install train inference deploy clean ui setup-ollama

help:
	@echo "Italian LLM - Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make train         - Start training"
	@echo "  make inference     - Start inference server"
	@echo "  make ui            - Open local UI"
	@echo "  make setup-ollama  - Install and pull Ollama model"
	@echo "  make deploy        - Deploy with Docker"
	@echo "  make clean         - Clean temporary files"
	@echo "  make data          - Download datasets"

install:
	pip install -r requirements.txt

train:
	python training/scripts/train_sft.py

inference:
	uvicorn api.main:app --reload --port 8000

ui:
	open /Users/jeaders/Desktop/myLLM/frontend/index.html

setup-ollama:
	@echo "Installing Ollama..."
	brew install --cask ollama
	@echo "Pulling llama3.1:8b model..."
	ollama pull llama3.1:8b
	@echo "Done. Now set USE_OLLAMA=true in .env and run 'make inference'"

deploy:
	docker-compose up --build

data:
	python data/download_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf logs/*.log
	rm -rf .pytest_cache

test:
	pytest tests/ -v

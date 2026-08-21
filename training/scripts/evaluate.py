#!/usr/bin/env python3
"""
Valutazione del modello italiano: perplexity, BLEU, BERTScore, factuality, toxicity.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load

logger = logging.getLogger(__name__)


class ItalianEvaluator:
    def __init__(self, model_path: str, tokenizer_name: str = None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bleu = load("bleu")
        self.bertscore = load("bertscore")
        self.perplexity = load("perplexity")

    def compute_perplexity(self, texts: List[str]) -> float:
        logger.info("Computing perplexity...")
        results = self.perplexity.compute(predictions=texts, model_id=self.tokenizer.name_or_path)
        return results["mean_perplexity"]

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        logger.info("Computing BLEU...")
        results = self.bleu.compute(predictions=predictions, references=references)
        return results["bleu"]

    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict:
        logger.info("Computing BERTScore...")
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="it",
            verbose=True,
        )
        return {
            "precision": sum(results["precision"]) / len(results["precision"]),
            "recall": sum(results["recall"]) / len(results["recall"]),
            "f1": sum(results["f1"]) / len(results["f1"]),
        }

    def compute_toxicity(self, texts: List[str]) -> Dict:
        logger.info("Computing toxicity...")
        try:
            from detoxify import Detoxify
            detector = Detoxify("original")
            scores = [detector.predict(t) for t in texts]
            avg_toxicity = sum(s["toxicity"] for s in scores) / len(scores)
            return {"avg_toxicity": avg_toxicity, "max_toxicity": max(s["toxicity"] for s in scores)}
        except Exception as e:
            logger.warning(f"Toxicity detection failed: {e}")
            return {"error": str(e)}

    def evaluate(self, test_data_path: str) -> Dict:
        logger.info(f"Loading test data from {test_data_path}")
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        predictions = []
        references = []

        for item in test_data[:100]:  # Limit to 100 for speed
            prompt = f"### Istruzione:\n{item['instruction']}\n\n### Risposta:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128, pad_token_id=self.tokenizer.eos_token_id)
            pred = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.append(pred)
            references.append([item["output"]])

        results = {
            "perplexity": self.compute_perplexity([d["output"] for d in test_data[:100]]),
            "bleu": self.compute_bleu(predictions, references),
            "bertscore": self.compute_bertscore(predictions, [r[0] for r in references]),
            "toxicity": self.compute_toxicity(predictions),
        }

        logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")
        return results


def main():
    evaluator = ItalianEvaluator("./models/merged/italian-llm-7b")
    results = evaluator.evaluate("./data/processed/test_set.json")
    with open("./training/logs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

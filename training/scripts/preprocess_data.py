#!/usr/bin/env python3
"""
Preprocessing del dataset italiano per il training.
Scarica, pulisce e prepara i dati per il fine-tuning.
"""
import json
import logging
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import requests
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from langdetect import detect_langs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    output_dir: Path
    max_samples: int = 10000
    min_text_length: int = 50
    max_text_length: int = 2048
    italian_threshold: float = 0.9
    deduplicate: bool = True
    tokenizer_name: str = "mistralai/Mistral-7B-v0.3"


class ItalianDataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.seen_hashes = set()

    def is_italian(self, text: str) -> bool:
        try:
            langs = detect_langs(text)
            italian_prob = sum(lang.prob for lang in langs if lang.lang == "it")
            return italian_prob >= self.config.italian_threshold
        except Exception:
            return False

    def dedup_hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:'\"àèéìíîòóùúäëïöüãõñçÀÈÉÌÍÎÒÓÙÚÄËÏÖÜÃÕÑÇ\-/]", "", text)
        return text.strip()

    def filter_text(self, text: str) -> bool:
        if not text or len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False
        if not self.is_italian(text):
            return False
        if self.config.deduplicate:
            h = self.dedup_hash(text)
            if h in self.seen_hashes:
                return False
            self.seen_hashes.add(h)
        return True

    def process_wikipedia(self) -> List[Dict]:
        logger.info("Loading Italian Wikipedia...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "it", split="train", streaming=True)
            texts = []
            for item in ds:
                text = self.clean_text(item.get("text", ""))
                if self.filter_text(text):
                    texts.append({"text": text, "source": "wikipedia", "title": item.get("title", "")})
                if len(texts) >= self.config.max_samples:
                    break
            logger.info(f"Collected {len(texts)} Wikipedia articles")
            return texts
        except Exception as e:
            logger.error(f"Wikipedia loading failed: {e}")
            return []

    def process_oscar(self) -> List[Dict]:
        logger.info("Loading OSCAR Italian...")
        try:
            ds = load_dataset("oscar", "unshuffled_deduplicated_it", split="train", streaming=True)
            texts = []
            for item in ds:
                text = self.clean_text(item.get("text", ""))
                if self.filter_text(text):
                    texts.append({"text": text, "source": "oscar"})
                if len(texts) >= self.config.max_samples:
                    break
            logger.info(f"Collected {len(texts)} OSCAR samples")
            return texts
        except Exception as e:
            logger.error(f"OSCAR loading failed: {e}")
            return []

    def process_culturax(self) -> List[Dict]:
        logger.info("Loading CulturaX Italian...")
        try:
            ds = load_dataset("uonlp/CulturaX", "it", split="train", streaming=True)
            texts = []
            for item in ds:
                text = self.clean_text(item.get("text", ""))
                if self.filter_text(text):
                    texts.append({"text": text, "source": "culturax"})
                if len(texts) >= self.config.max_samples:
                    break
            logger.info(f"Collected {len(texts)} CulturaX samples")
            return texts
        except Exception as e:
            logger.error(f"CulturaX loading failed: {e}")
            return []

    def save_dataset(self, samples: List[Dict], filename: str):
        output_file = self.output_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(samples)} samples to {output_file}")

    def run(self):
        logger.info("Starting data preprocessing...")
        all_samples = []

        # Load datasets
        all_samples.extend(self.process_wikipedia())
        all_samples.extend(self.process_oscar())
        all_samples.extend(self.process_culturax())

        # Deduplicate across sources
        seen = set()
        deduped = []
        for sample in all_samples:
            h = self.dedup_hash(sample["text"])
            if h not in seen:
                seen.add(h)
                deduped.append(sample)

        logger.info(f"Total unique samples: {len(deduped)}")

        # Save
        self.save_dataset(deduped, "italian_corpus.json")
        self.save_dataset(deduped[: self.config.max_samples], "italian_corpus_trimmed.json")
        logger.info("Preprocessing complete!")


def main():
    config = DataConfig(
        output_dir=Path("./data/processed"),
        max_samples=5000,
        italian_threshold=0.85,
    )
    preprocessor = ItalianDataPreprocessor(config)
    preprocessor.run()


if __name__ == "__main__":
    main()

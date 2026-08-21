#!/usr/bin/env python3
"""
Script per scaricare e preparare dataset italiani
"""
import requests
from pathlib import Path
import gzip
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: Path):
    """Download file with progress"""
    logger.info(f"Downloading {url} -> {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Saved to {dest_path}")


def download_italian_wikipedia(output_dir: Path):
    """Scarica dump Wikipedia italiano"""
    logger.info("Downloading Italian Wikipedia...")
    url = "https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2"
    dest = output_dir / "itwiki-latest-pages-articles.xml.bz2"
    download_file(url, dest)


def download_oscar_it(output_dir: Path):
    """Scarica dataset OSCAR italiano da HuggingFace"""
    logger.info("OSCAR dataset available on HuggingFace:")
    logger.info("https://huggingface.co/datasets/oscar")
    logger.info("Use: datasets.load_dataset('oscar', 'unshuffled_deduplicated_it')")


def download_legal_texts(output_dir: Path):
    """Scarica testi legali da giustizia.it"""
    logger.info("Downloading Italian legal texts...")
    # Implement web scraping for legal texts
    base_url = "https://www.giustizia.it"
    # Add specific URLs for laws, decrees, etc.
    pass


def create_sample_instructions(output_dir: Path):
    """Crea dataset di istruzioni italiane di esempio"""
    instructions = [
        {
            "instruction": "Spiega la differenza tra una repubblica e una monarchia",
            "input": "",
            "output": "Una repubblica è una forma di governo in cui il capo dello Stato non è un erede..."
        },
        # Aggiungi più istruzioni qui
    ]
    
    import json
    output_file = output_dir / "italian_instructions_sample.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    logger.info(f"Created sample instructions at {output_file}")


def main():
    output_dir = Path("./data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting data download...")
    
    # Download datasets
    download_italian_wikipedia(output_dir)
    download_oscar_it(output_dir)
    download_legal_texts(output_dir)
    create_sample_instructions(output_dir)
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    main()

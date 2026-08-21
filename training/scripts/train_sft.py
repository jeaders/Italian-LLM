import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model and tokenizer with quantization"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, config: dict):
    """Configure LoRA adapter"""
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def format_instruction(example):
    """Format dataset example for instruction tuning"""
    return {
        "text": f"### Istruzione:\n{example['instruction']}\n\n### Input:\n{example.get('input', '')}\n\n### Risposta:\n{example['output']}"
    }

def main():
    config = load_config("training/configs/sft_config.yaml")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    model = setup_lora(model, config)
    
    # Load and format dataset
    dataset = load_dataset(config["dataset_name"], split="train")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation", 4),
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=True,
        bf16=False,
        tf32=True,
        report_to="wandb" if config.get("use_wandb") else "none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=LoraConfig(**config["lora"]),
        max_seq_length=config.get("max_seq_length", 2048),
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save adapter
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    logger.info("Training complete!")

if __name__ == "__main__":
    main()

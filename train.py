from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def format_example(example):
    text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    return {"text": text}


def load_dataset_for_training(dataset_path):
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(format_example)
    return dataset


def tokenize(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    return tokenizer, model


def apply_qlora(model):
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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

    model = get_peft_model(model, peft_config)
    return model, peft_config


def train():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path = Path("data/dataset.jsonl")

    dataset = load_dataset_for_training(dataset_path)
    tokenizer, model = tokenize(model_name)
    model, peft_config = apply_qlora(model)

if __name__ == "__main__":
    train()

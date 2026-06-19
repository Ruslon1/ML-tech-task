from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def format_example(example):
    text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    return {"text": text}


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path = Path("data/dataset.jsonl")

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(format_example)

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

if __name__ == "__main__":
    main()

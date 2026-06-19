from pathlib import Path

import torch
from datasets import load_dataset

def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path = Path("data/dataset.jsonl")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")



if __name__ == "__main__":
    main()

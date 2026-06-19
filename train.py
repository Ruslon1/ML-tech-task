from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from tracking import log_artifacts, log_metrics, log_params, start_run


def format_example(example):
    text = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    return {"text": text}


def load_dataset_for_training(dataset_path):
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(format_example)
    return dataset


def tokenize(model_name, dataset):
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

    def tokenize_example(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)

    return tokenizer, model, dataset


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
    output_dir = Path("outputs/mistral-lora")
    adapter_dir = output_dir / "adapter"
    dataset = load_dataset_for_training(dataset_path)
    tokenizer, model, dataset = tokenize(model_name, dataset)
    model, peft_config = apply_qlora(model)

    output_dir.mkdir(parents=True, exist_ok=True)

    with start_run("mistral-qlora"):
        log_params(
            {
                "model_name": model_name,
                "train_samples": len(dataset),
                "num_train_epochs": 2,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "max_seq_length": 512,
                "warmup_steps": 20,
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
            }
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            warmup_steps=20,
            lr_scheduler_type="cosine",
            fp16=True,
            bf16=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        trainer.train()

        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        log_metrics(trainer)
        log_artifacts(output_dir)

if __name__ == "__main__":
    train()

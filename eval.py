from pathlib import Path

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, instruction):
    prompt = f"<s>[INST] {instruction} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("[/INST]", 1)[-1].strip()


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path = Path("outputs/mistral-lora/adapter")
    evaluation_path = Path("data/evaluation.jsonl")
    output_path = Path("outputs/eval_results.csv")

    dataset = load_dataset("json", data_files=str(evaluation_path), split="train")

    base_tokenizer, base_model = load_model(model_name)
    tuned_tokenizer, tuned_model = load_model(model_name, adapter_path)

    rows = []
    references = []
    base_predictions = []
    tuned_predictions = []

    for example in dataset:
        instruction = example["instruction"]
        reference = example["response"]

        base_response = generate_response(base_tokenizer, base_model, instruction)
        tuned_response = generate_response(tuned_tokenizer, tuned_model, instruction)

        references.append(reference)
        base_predictions.append(base_response)
        tuned_predictions.append(tuned_response)

        rows.append(
            {
                "instruction": instruction,
                "reference_response": reference,
                "base_response": base_response,
                "tuned_response": tuned_response,
            }
        )

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    base_rouge = rouge.compute(predictions=base_predictions, references=references)
    tuned_rouge = rouge.compute(predictions=tuned_predictions, references=references)

    base_bertscore = bertscore.compute(
        predictions=base_predictions,
        references=references,
        lang="en",
    )
    tuned_bertscore = bertscore.compute(
        predictions=tuned_predictions,
        references=references,
        lang="en",
    )

    results_df = pd.DataFrame(rows)
    results_df["base_bertscore_f1"] = base_bertscore["f1"]
    results_df["tuned_bertscore_f1"] = tuned_bertscore["f1"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

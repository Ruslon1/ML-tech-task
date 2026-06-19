from pathlib import Path

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
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


def generate_predictions(dataset, model_name, adapter_path=None):
    tokenizer, model = load_model(model_name, adapter_path)
    predictions = []

    for example in dataset:
        predictions.append(generate_response(tokenizer, model, example["instruction"]))

    del model
    torch.cuda.empty_cache()

    return predictions


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path = Path("outputs/mistral-lora/adapter")
    evaluation_path = Path("data/evaluation.jsonl")
    output_path = Path("outputs/eval_results.csv")

    dataset = load_dataset("json", data_files=str(evaluation_path), split="train")
    references = [example["response"] for example in dataset]

    base_predictions = generate_predictions(dataset, model_name)
    tuned_predictions = generate_predictions(dataset, model_name, adapter_path)

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

    rows = []
    for i, example in enumerate(dataset):
        rows.append(
            {
                "instruction": example["instruction"],
                "reference_response": example["response"],
                "base_response": base_predictions[i],
                "tuned_response": tuned_predictions[i],
                "base_bertscore_f1": base_bertscore["f1"][i],
                "tuned_bertscore_f1": tuned_bertscore["f1"][i],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)



if __name__ == "__main__":
    main()

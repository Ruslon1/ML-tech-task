from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv(
        "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )

    df = df[df["category"] == "CANCEL"].copy()
    df = df[["instruction", "response"]].copy()

    df = df.dropna(subset=["instruction", "response"])

    df["instruction"] = df["instruction"].astype(str).apply(lambda x: " ".join(x.split()))
    df["response"] = df["response"].astype(str).apply(lambda x: " ".join(x.split()))

    df = df[
        (df["instruction"].str.len() > 0)
        & (df["response"].str.len() > 0)
    ].copy()

    df = df.drop_duplicates(subset=["instruction", "response"]).reset_index(drop=True)

    output_path = Path("data/dataset.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

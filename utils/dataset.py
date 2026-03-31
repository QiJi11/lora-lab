from datasets import load_dataset
from transformers import AutoTokenizer


def load_jsonl_dataset(data_path: str):
    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset


def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    def tokenize(example):
        prompt = example.get("instruction", "")
        response = example.get("output", "")
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, remove_columns=dataset.column_names)

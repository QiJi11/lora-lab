from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "./models/lora-output"
    data_path: str = "./data/train.jsonl"
    max_length: int = 512
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True

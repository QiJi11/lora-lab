# lora-lab

QLoRA fine-tuning project based on transformers + PEFT.

## Requirements
- Python 3.10+
- CUDA 11.8+ (T4/A100 recommended)
- Windows/Linux

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Prepare training data (JSONL format)
# Each line: {"instruction": "question", "output": "answer"}

python train.py
```

## LoRA Key Parameters
- `lora_r=16`: rank of low-rank matrices, higher = more params, better effect
- `lora_alpha=32`: scaling factor, typically = 2*r
- `target_modules`: only fine-tune q/v matrices in attention layers
- `load_in_4bit=True`: QLoRA 4-bit quantization, reduces GPU memory ~75%

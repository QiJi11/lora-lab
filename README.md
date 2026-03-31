# lora-lab

**QLoRA 大模型指令微调实验室**

> 技术栈：Python + PEFT + Transformers + BitsAndBytes + TRL (SFTTrainer)

## 项目架构

```
lora-lab/
├── train.py           # 微调入口（一键运行）
├── config.py          # 超参配置（dataclass）
├── requirements.txt   # 依赖（含 PyTorch + PEFT + TRL）
├── utils/
│   └── dataset.py     # JSONL 数据加载 + Instruction 格式 tokenize
└── data/
    └── train.jsonl    # 训练数据（instruction/output 格式）
```

## 核心特性

### 1. QLoRA 4-bit 量化（低显存微调）
- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- 显存降低约 75%（7B 模型从 ~28GB → ~7GB）
- 适合 T4/RTX 3090 等消费级 GPU

### 2. LoRA 低秩矩阵（参数高效）
```python
LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
```
- 只训练注意力层的低秩分解矩阵 A·B
- 可训练参数 < 0.1%，效果接近全参微调
- 原始权重冻结，不破坏预训练知识

### 3. Instruction 微调格式
```
### Instruction:
{问题}

### Response:
{答案}
```
- 适合 SFT（Supervised Fine-Tuning）
- 兼容 Qwen/LLaMA/Mistral 等主流模型

### 4. 默认模型：Qwen2.5-0.5B-Instruct
- 0.5B 小模型，本地 CPU 可跑（速度慢）
- 换成 Qwen2.5-7B 或 LLaMA-3-8B 效果更好

## 快速运行

```bash
.\venv\Scripts\activate

# 准备数据（JSONL 格式）
# data/train.jsonl 已有 5 条示例

# 开始微调
python train.py
```

## 超参说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `lora_r` | 16 | LoRA 秩，越大参数越多效果越好 |
| `lora_alpha` | 32 | 缩放系数，通常 = 2×r |
| `load_in_4bit` | True | QLoRA 4bit 量化 |
| `num_epochs` | 3 | 训练轮次 |
| `learning_rate` | 2e-4 | 学习率 |
| `batch_size` | 4 | 批大小（显存不够就调小） |

## 数据格式

```json
{"instruction": "什么是RAG？", "output": "RAG是检索增强生成..."}
```

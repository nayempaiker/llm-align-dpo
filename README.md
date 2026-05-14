# llm-align-dpo

Fine-tuning Mistral-7B end-to-end with a two-stage DPO alignment pipeline on a single RTX 3090.

## Results

| Model | LLM-judge score | vs base |
|---|---|---|
| Mistral-7B base | 7.3 / 10 | — |
| + SFT | 8.9 / 10 | +21.9% |
| + DPO | 9.2 / 10 | +26.0% |

DPO win rate vs SFT: **71%** — averaged across three evaluation runs of 50 prompts each, judged by GPT-4o-mini.

Model weights on **HuggingFace**: [npaiker/mistral-7b-dpo-ultrafeedback](https://huggingface.co/npaiker/mistral-7b-dpo-ultrafeedback)

Training runs on **Wandb**: [exp_001 report](https://api.wandb.ai/links/nayempaiker-labs/p3zx1xqw)

## Pipeline

```
Mistral-7B-v0.1 (base)
        ↓
   SFT with QLoRA          59,510 instruction examples from UltraFeedback
        ↓
   DPO training            47,731 preference pairs (score gap ≥ 1.0)
        ↓
   Evaluation              GPT-4o-mini as judge, 50 prompts × 3 runs
```

## Training details

**SFT — Stage 1**

| | |
|---|---|
| Dataset | `openbmb/UltraFeedback` chosen responses |
| Samples | 59,510 |
| Method | QLoRA — 4-bit NF4, LoRA r=16 alpha=32 |
| LR | 2e-4, cosine schedule |
| Epochs | 1 |
| Final train loss | 0.990 |
| Final eval loss | 0.977 |
| Training time | 15h 44m |

**DPO — Stage 2**

| | |
|---|---|
| Dataset | `openbmb/UltraFeedback` preference pairs |
| Pairs | 47,731 (filtered, score gap ≥ 1.0) |
| β | 0.1 |
| LR | 5e-5, cosine schedule |
| Epochs | 1 |
| Final train loss | 0.195 |
| Final eval loss | 0.182 |
| Reward margin | 12.54 |
| Reward accuracy | 91.6% |
| Training time | 29h 28m |

## Quickstart

```bash
git clone https://github.com/nayempaiker/llm-align-dpo
cd llm-align-dpo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# prepare data
python scripts/prepare_data.py --dataset ultrafeedback

# train
python scripts/train_sft.py --exp exp_001 --dataset ultrafeedback_v1
python scripts/train_dpo.py --exp exp_001 --dataset ultrafeedback_v1

# evaluate
python scripts/run_eval.py --exp exp_001 --n-prompts 50
```

## Load the model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "npaiker/mistral-7b-dpo-ultrafeedback")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
```

## Project structure

```
llm-align-dpo/
├── src/
│   ├── data/
│   │   ├── base_dataset.py       
│   │   ├── dataset_ultrafeedback.py
│   │   ├── dataset_openhermes.py
│   │   └── registry.py
│   └── training/
│       ├── sft_trainer.py
│       └── dpo_trainer.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── run_eval.py
│   ├── demo.py
│   └── push_to_hub.py
├── data/processed/ultrafeedback_v1/
├── results/exp_001/
```

## Experiments

| Exp | SFT data | DPO data | Score | Win rate | Status |
|---|---|---|---|---|---|
| exp_001 | UltraFeedback 59k | UltraFeedback 47k | 9.2/10 | 71% | ✓ |

## Stack

- `mistralai/Mistral-7B-v0.1` — base model
- `bitsandbytes` — 4-bit NF4 quantization
- `peft` — LoRA adapters
- `trl` 1.0 — SFTTrainer and DPOTrainer
- `openbmb/UltraFeedback` — preference dataset
- W&B — experiment tracking
- GPT-4o-mini — evaluation judge

## Hardware

RTX 3090 24GB
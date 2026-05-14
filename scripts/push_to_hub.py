from __future__ import annotations
 
import argparse
import logging
import os
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
 
from huggingface_hub import HfApi, create_repo
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
# Read from .env
HF_USERNAME = os.getenv("HF_USERNAME", "HUGGINGFACE_USERNAME")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "nayempaiker")
HF_TOKEN    = os.getenv("HF_TOKEN", None)
 
 
# ------------------------------------------------------------------ #
# Model card                                                           #
# ------------------------------------------------------------------ #
 
MODEL_CARD = f"""---
language:
- en
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.1
tags:
- mistral
- dpo
- rlhf
- alignment
- peft
- lora
- qlora
datasets:
- openbmb/UltraFeedback
---
 
# Mistral-7B DPO — UltraFeedback
 
Mistral-7B fine-tuned end-to-end with a two-stage DPO alignment pipeline on a single RTX 3090.
 
## Results
 
| Model | LLM-judge score |
|---|---|
| Mistral-7B base | 7.3 / 10 |
| Mistral-7B SFT | 8.9 / 10 |
| **Mistral-7B DPO (this model)** | **9.1 / 10** |
 
**DPO win rate vs SFT: 73%** across two evaluation runs of 50 prompts each, judged by GPT-4o-mini.
 
## Training
 
**Stage 1 — SFT:**
- Dataset: `openbmb/UltraFeedback` chosen responses (59,510 examples)
- Base model: `mistralai/Mistral-7B-v0.1`
- Method: QLoRA (4-bit NF4, LoRA r=16, alpha=32)
- Final train loss: 0.990 | Eval loss: 0.977
- Training time: 15h 44m on RTX 3090
 
**Stage 2 — DPO:**
- Dataset: `openbmb/UltraFeedback` preference pairs (47,731 pairs, score gap ≥ 1.0)
- β: 0.1
- Final train loss: 0.195 | Eval loss: 0.182
- Reward margin: 12.54 | Reward accuracy: 91.6%
- Training time: 29h 28m on RTX 3090
 
## Usage
 
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
 
base_model = "mistralai/Mistral-7B-v0.1"
adapter    = "{HF_USERNAME}/mistral-7b-dpo-ultrafeedback"
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
 
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, adapter)
tokenizer = AutoTokenizer.from_pretrained(base_model)
 
prompt = "<|im_start|>user\\nExplain recursion to a 10-year-old.<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
 
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
 
## Hardware
 
- GPU: NVIDIA RTX 3090 24GB
- RAM: 32GB
- VRAM usage: ~8-10GB during training (QLoRA)
 
## Code
 
Full training pipeline: [github.com/{GITHUB_USERNAME}/llm-align-dpo](https://github.com/{GITHUB_USERNAME}/llm-align-dpo)
"""
 
 
# ------------------------------------------------------------------ #
# Push                                                                 #
# ------------------------------------------------------------------ #
 
def push_to_hub(exp: str, repo_id: str, private: bool = False) -> None:
    dpo_checkpoint = PROJECT_ROOT / "checkpoints" / exp / "dpo" / "final"
 
    if not dpo_checkpoint.exists():
        logger.error(f"DPO checkpoint not found: {dpo_checkpoint}")
        sys.exit(1)
 
    if HF_USERNAME == "HUGGINGFACE_USERNAME":
        logger.error("HF_USERNAME not set in .env — add HF_USERNAME=your_username")
        sys.exit(1)
 
    logger.info(f"Pushing {dpo_checkpoint} to {repo_id}...")
 
    api = HfApi(token=HF_TOKEN)
 
    # Create repo if it doesn't exist
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=HF_TOKEN,
    )
    logger.info(f"Repo ready: {repo_id}")
 
    # Write model card into checkpoint folder
    card_path = dpo_checkpoint / "README.md"
    with open(card_path, "w") as f:
        f.write(MODEL_CARD)
    logger.info(f"Model card written -> {card_path}")
 
    # Upload checkpoint folder
    api.upload_folder(
        folder_path=str(dpo_checkpoint),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload DPO checkpoint — {exp}",
    )
 
    logger.info(f"Done! Model live at: https://huggingface.co/{repo_id}")
 
 
# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #
 
def parse_args():
    default_repo = f"{HF_USERNAME}/mistral-7b-dpo-ultrafeedback"
 
    parser = argparse.ArgumentParser(
        description="Push DPO checkpoint to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name, e.g. exp_001.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=default_repo,
        help="HuggingFace repo id. Defaults to HF_USERNAME/mistral-7b-dpo-ultrafeedback from .env.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private.",
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    push_to_hub(
        exp=args.exp,
        repo_id=args.repo,
        private=args.private,
    )
 

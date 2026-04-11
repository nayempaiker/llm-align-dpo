# llm-align-dpo
 
Fine-tuning Mistral-7B end-to-end with a DPO alignment pipeline on a consumer gpu.
 
## What this project does
 
Takes a raw pretrained Mistral-7B and makes it meaningfully more helpful and aligned with human preferences through two training stages:
 
1. **SFT** — supervised fine-tuning on 59,510 high-quality instruction examples from UltraFeedback
2. **DPO** — direct preference optimization on 47,731 human preference pairs (chosen vs rejected responses)
 
## Training progress
 
Training curves and metrics:
**[W&B Report — exp_001 results](https://api.wandb.ai/links/nayempaiker-labs/p3zx1xqw)**
 
### SFT results (exp_001_sft)
 
| Metric | Start | End |
|---|---|---|
| Train loss | 1.265 | 0.990 |
| Eval loss | — | 0.977 |
| Token accuracy | 69.2% | 73.2% |
| Training time | — | 15h 44m |
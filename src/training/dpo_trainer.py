"""
Direct Preference Optimization fine-tuning on top of the SFT checkpoint

"""

from __future__ import annotations
import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig



logger = logging.getLogger(__name__)

# set config

@dataclass
class DPOTrainingConfig:
    # model 
    base_model: str = "mistralai/Mistral-7B-v0.1"

    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


    # LoRA 
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # DPO-specific
    beta: float = 0.1 
    max_prompt_length: int = 512
    max_length: int = 1024 

    # training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2  
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4 
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 0.01


    # eval and saving
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
 
    # logging
    logging_steps: int = 10
    report_to: str = "wandb"
 
    # misc
    seed: int = 42
    dataloader_num_workers: int = 2
    fp16: bool = False
    bf16: bool = True



# load model
def load_model_for_dpo(sft_checkpoint_dir: Path, cfg: DPOTrainingConfig):
    """
    Load SFT checkpoint as policy model, which is a LoRA adapter saved on top of the base model
    """
    logger.info(f"Loading policy model from SFT checkpoint: {sft_checkpoint_dir}")

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    # load base model quantized
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )


    base_model = prepare_model_for_kbit_training(base_model)

    # load sft LoRA adapter on top
    model = PeftModel.from_pretrained(
        base_model,
        str(sft_checkpoint_dir),
        is_trainable=True  # weights get updated during training
    )

    logger.info(f"Policy model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model



def load_reference_model(sft_checkpoint_dir: Path, cfg:DPOTrainingConfig):
    """
    Load the same SFT cehckpoint as the frozen reference model
    """
    logger.info(f"loading reference model (frozen)")

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    # load the base model quantized
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )

    # load sft adapter
    ref_model = PeftModel.from_pretrained(
        base_model,
        str(sft_checkpoint_dir),
        is_trainable=False,     # frozen
    )
 
    logger.info("Reference model loaded and frozen.")
    return ref_model


# tokenizer
def load_tokenizer(cfg: DPOTrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code = True
    )

    # set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # set left padding (ensures the actual response tokens are at the end of the sequence)
    tokenizer.padding_side = "left"


    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")

    return tokenizer




# dataset loading
def load_dpo_dataset(
        processed_dir: Path,
        max_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load dpo_train.jsonl and dpo_eval.jsonl
    """
    def load_jsonl(path: Path) -> list[dict]:
        """
        Loads the jsonl files
        """
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
 
    def format_sample(row: dict) -> dict:
        """
        Format into ChatML strings.
        DPOTrainer format:
            prompt: the user turn only
            chosen: the full preferred response
            rejected: the full dis-preferred response
        """
        prompt = f"<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        chosen   = row["chosen"] + "<|im_end|>"
        rejected = row["rejected"] + "<|im_end|>"


        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    logger.info(f"Loading DPO data from {processed_dir}")
 
    train_raw = load_jsonl(processed_dir / "dpo_train.jsonl")
    eval_raw  = load_jsonl(processed_dir / "dpo_eval.jsonl")
 
    # cap eval to 500 (same as sft)
    eval_raw = eval_raw[:500]
 
    if max_samples:
        train_raw = train_raw[:max_samples]
        logger.info(f"Capped DPO training set to {max_samples} samples")
 
    train_dataset = Dataset.from_list([format_sample(r) for r in train_raw])
    eval_dataset  = Dataset.from_list([format_sample(r) for r in eval_raw])
 
    logger.info(f"DPO Train: {len(train_dataset):,} pairs")
    logger.info(f"DPO Eval:  {len(eval_dataset):,} pairs")
 
    return train_dataset, eval_dataset


# setup trainer
def build_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: Path,
    cfg: DPOTrainingConfig,
    run_name: str,
) -> DPOTrainer:
    """
    Assemble DPO trainer
    """
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
 
        # DPO-specific
        beta=cfg.beta,
        # max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
 

        # training
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
 

        # precision
        fp16=cfg.fp16,
        bf16=cfg.bf16,
 

        # Eval and saving
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
 

        # logging
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
 

        # misc
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
    )


    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
 
    return trainer


# main
def run_dpo(
    sft_checkpoint_dir: Path,
    processed_dir: Path,
    checkpoint_dir: Path,
    exp_name: str,
    cfg: DPOTrainingConfig | None = None,
    wandb_project: str = "llm-align-dpo",
    max_samples: int | None = None,
) -> Path:
    """
    Full DPO training run.
 
    Args:
        sft_checkpoint_dir: Path to checkpoints/<exp>/sft/final/
        processed_dir: Path to data/processed/ultrafeedback_v1/
        checkpoint_dir: Path to checkpoints/<exp>/dpo/
        exp_name: wandb run name, e.g. "exp_001_dpo"
        cfg: DPOTrainingConfig — uses defaults if None
        wandb_project: wandb project name
        max_samples: cap on DPO training pairs (None = use all)
 
    Returns:
        Path to the saved DPO checkpoint
    """

    if cfg is None:
        cfg = DPOTrainingConfig()
 
    # wandb setup
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = exp_name
 
    logger.info(f"Starting DPO run: {exp_name}")
    logger.info(f"SFT checkpoint: {sft_checkpoint_dir}")
    logger.info(f"Processed data: {processed_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Beta: {cfg.beta}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
 
    # load models
    model = load_model_for_dpo(sft_checkpoint_dir, cfg)
    ref_model = load_reference_model(sft_checkpoint_dir, cfg)
    tokenizer = load_tokenizer(cfg)
 

    # load data
    train_dataset, eval_dataset = load_dpo_dataset(
        processed_dir=processed_dir,
        max_samples=max_samples,
    )
 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 

    # build and run trainer
    trainer = build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=checkpoint_dir,
        cfg=cfg,
        run_name=exp_name,
    )
 
    logger.info("Starting DPO training...")
    trainer.train()
 
    # save final adapter
    final_dir = checkpoint_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"DPO adapter saved -> {final_dir}")
 
    return final_dir












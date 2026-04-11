from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

logger = logging.getLogger(__name__)


# all hyperparams
@dataclass
class TrainingConfig:
    # model
    base_model: str = "mistralai/Mistral-7B-v0.1"

    # qlora quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"           
    bnb_4bit_compute_dtype: str = "bfloat16" 
    bnb_4bit_use_double_quant: bool = True

    # lora adapter
    lora_r: int = 16   
    lora_alpha: int = 32    
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",   
            "gate_proj", "up_proj", "down_proj",      
        ]
    )

    # training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8   #4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2   #4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 1024

    # eval and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500   # 100
    save_strategy: str = "steps"
    save_steps: int = 500   # 100
    save_total_limit: int = 2                  
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # logging
    logging_steps: int = 10
    report_to: str = "wandb"

    # misc
    seed: int = 42
    dataloader_num_workers: int = 4
    fp16: bool = False
    bf16: bool = True  



# load model
def load_base_model(cfg: TrainingConfig):
    """
    Load Mistral-7B in 4-bit with bitsandbytes QLoRA config
    """
    logger.info(f"Loading base model: {cfg.base_model}")

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = cfg.load_in_4bit,
        bnb_4bit_quant_type = cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant = cfg.bnb_4bit_use_double_quant
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config = bnb_config,
        device_map = "auto",
        trust_remote_code = True,
        torch_dtype=compute_dtype

    )

    # required before adding lora to quantized model
    model = prepare_model_for_kbit_training(model)

    logger.info(f"Base model loaded. Parameters: {model.num_parameters():,}")
    return model

def attach_lora(model, cfg:TrainingConfig):
    """
    Attach lora adapter to the quantized base model
    """
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
 
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  
    return model


# tokenizer
def load_tokenizer(cfg: TrainingConfig):
    """
    Load tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )

    # mistral doesn't have a pad token by default — use eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
 
    # pad on the right for causal LM training
    tokenizer.padding_side = "right"
 
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    return tokenizer


# dataset loading
def load_sft_dataset(
        processed_dir: Path,
        tokenizer,
        cfg: TrainingConfig,
        max_samples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load sft_train.jsonl and sft_eval.jsonl from the processed directory.
    Formats each sample as a ChatML string for the trainer.
    """

    def load_jsonl(path: Path) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def format_sample(row: dict) -> dict:
        """Convert raw SFTSample dict to ChatML formatted text."""
        text = (
            f"<|im_start|>user\n{row['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{row['output']}<|im_end|>"
        )
        return {"text": text}
    
    logger.info(f"Loading SFT data from {processed_dir}")
 
    train_raw = load_jsonl(processed_dir / "sft_train.jsonl")
    eval_raw  = load_jsonl(processed_dir / "sft_eval.jsonl")
    eval_raw  = eval_raw[:500] 
 
    if max_samples:
        train_raw = train_raw[:max_samples]
        logger.info(f"Capped training set to {max_samples} samples")
 
    train_dataset = Dataset.from_list([format_sample(r) for r in train_raw])
    eval_dataset  = Dataset.from_list([format_sample(r) for r in eval_raw])
 
    logger.info(f"Train: {len(train_dataset):,} samples")
    logger.info(f"Eval:  {len(eval_dataset):,} samples")
 
    return train_dataset, eval_dataset


# trainer setup
def build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: Path,
    cfg: TrainingConfig,
    run_name: str,
) -> SFTTrainer:
    """
    Assemble SFTTrainer with all components.
    tokenizer, max_seq_length, and dataset_text_field go into TRLSFTConfig.
    """

    # SFTConfig extends TrainingArguments and adds SFT-specific args
    training_args = TRLSFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,

        # SFT-specific 
        max_length=cfg.max_seq_length,
        dataset_text_field="text",

        # training
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=100,   
        weight_decay=cfg.weight_decay,

        # precision
        fp16=cfg.fp16,
        bf16=cfg.bf16,

        # eval and saving
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

    # tokenizer passed via processing_class, not tokenizer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    return trainer


# main
def run_sft(
    processed_dir: Path,
    checkpoint_dir: Path,
    exp_name: str,
    cfg: TrainingConfig | None = None,
    wandb_project: str = "llm-align-dpo",
    max_samples: int | None = None,
) -> Path:
    """
    Full SFT training run.
 
    Args:
        processed_dir: Path to ultrafeedback
        checkpoint_dir: Path to sft checkpoints
        exp_name: Experiment name for wandb run
        cfg: TrainingConfig
        wandb_project:  wandb project name
 
    Returns:
        Path to the saved SFT checkpoint
    """
    if cfg is None:
        cfg = TrainingConfig()
 
    # wandb setup
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = exp_name
 
    logger.info(f"Starting SFT run: {exp_name}")
    logger.info(f"Processed data: {processed_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
 
    # load everything
    model     = load_base_model(cfg)
    model     = attach_lora(model, cfg)
    tokenizer = load_tokenizer(cfg)
 
    train_dataset, eval_dataset = load_sft_dataset(
        processed_dir=processed_dir,
        tokenizer=tokenizer,
        cfg=cfg,
        max_samples=max_samples,
    )
 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=checkpoint_dir,
        cfg=cfg,
        run_name=exp_name,
    )
 
    # train
    logger.info("Starting training...")
    trainer.train()
 
    # save final adapter
    final_dir = checkpoint_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"SFT adapter saved -> {final_dir}")
 
    return final_dir
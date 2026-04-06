"""
CLI entrypoint for the SFT training
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
 
from src.training.sft_trainer import TrainingConfig as SFTConfig, run_sft

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT fine-tuning on a processed dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # required
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name, e.g. exp_001. Used for checkpoint dir and W&B run name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Processed dataset folder name, e.g. ultrafeedback_v1.",
    )
 
    # model
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model ID for the base model.",
    )
 
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
 

    # training
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples. None = use all.")
 

    # logging
    parser.add_argument("--wandb-project", type=str, default="llm-align-dpo", help="W&B project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
 
    # misc
    parser.add_argument("--dry-run", action="store_true", help="Print config without training.")
 
    return parser.parse_args()


# set path
def build_paths(exp: str, dataset: str) -> tuple[Path, Path]:
    """
    processed_dir: data/processed/<dataset>/
    checkpoint_dir: checkpoints/<exp>/sft/
    """
    processed_dir  = PROJECT_ROOT / "data" / "processed" / dataset
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / exp / "sft"
    return processed_dir, checkpoint_dir


# main
def main() -> None:
    args = parse_args()
 
    processed_dir, checkpoint_dir = build_paths(args.exp, args.dataset)
 
    # validate processed dir exists
    if not processed_dir.exists():
        logger.error(
            f"Processed dataset not found: {processed_dir}\n"
            f"Run: python scripts/prepare_data.py --dataset <name> first."
        )
        sys.exit(1)
 

    # validate SFT train file exists
    if not (processed_dir / "sft_train.jsonl").exists():
        logger.error(f"sft_train.jsonl not found in {processed_dir}")
        sys.exit(1)
 

    # build config from args
    cfg = SFTConfig(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        report_to="none" if args.no_wandb else "wandb",
    )
 
    exp_name = f"{args.exp}_sft"
 
    
    # dry run
    if args.dry_run:
        print("\n--- DRY RUN ---\n")
        print(f"exp: {args.exp}")
        print(f"dataset: {args.dataset}")
        print(f"processed_dir: {processed_dir}")
        print(f"checkpoint_dir: {checkpoint_dir}")
        print(f"base_model: {cfg.base_model}")
        print(f"lora_r: {cfg.lora_r}")
        print(f"lora_alpha: {cfg.lora_alpha}")
        print(f"epochs: {cfg.num_train_epochs}")
        print(f"batch_size: {cfg.per_device_train_batch_size}")
        print(f"grad_accum: {cfg.gradient_accumulation_steps}")
        print(f"effective_batch: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
        print(f"learning_rate: {cfg.learning_rate}")
        print(f"max_seq_length: {cfg.max_seq_length}")
        print(f"max_samples: {args.max_samples or 'all'}")
        print(f"wandb_project: {args.wandb_project}")
        print(f"W&B logging: {'disabled' if args.no_wandb else 'enabled'}")
        print("\nNo training started. Remove --dry-run to execute.\n")
        return
 

    # main run
    logger.info(f"Experiment: {args.exp}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Base model: {cfg.base_model}")
    logger.info(f"Epochs: {cfg.num_train_epochs}")
    logger.info(f"Effective batch: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"Max seq length: {cfg.max_seq_length}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
 
    final_dir = run_sft(
        processed_dir=processed_dir,
        checkpoint_dir=checkpoint_dir,
        exp_name=exp_name,
        cfg=cfg,
        wandb_project=args.wandb_project,
        max_samples=args.max_samples,
    )
 
    print("\n" + "=" * 50)
    print("SFT Training done")
    print("=" * 50)
    print(f"Experiment: {args.exp}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {final_dir}")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()

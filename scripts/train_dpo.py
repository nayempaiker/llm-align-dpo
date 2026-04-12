"""
CLI endpoint for DPO training
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.training.dpo_trainer import DPOTrainingConfig, run_dpo



# logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger  = logging.getLogger(__name__)


# args parsing
def parse_args() -> argparse.Namespace:
    """
    Command Line Args:
        --exp: experiment name/number
        --dataset: dataset name
        --beta: KL penalty strength
        --max-prompt-len: token's max prompt length
        --max-len: Max total sequence length
        --epochs: number of epochs
        --batch-size: batch size per device
        --grad-accum: gradient accumulation steps
        --lr: learning rate
        --max-samples: max samples
        --wandb-project: wandb project name
        --no-wandb: disable wandb logging
        --dry-run: Print config without training 
    """

    parser = argparse.ArgumentParser(
        description="Run DPO training on top of an SFT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # required
    parser.add_argument("--exp", type=str, required=True, help="Experiment name, e.g. exp_001.")
    parser.add_argument("--dataset", type=str, required=True, help="Processed dataset folder, e.g. ultrafeedback_v1.")
 
    # DPO-specific
    parser.add_argument("--beta", type=float, default=0.1,  help="KL penalty. Lower = more aggressive preference learning.")
    parser.add_argument("--max-prompt-len", type=int,   default=512,  help="Max prompt length in tokens.")
    parser.add_argument("--max-len", type=int,   default=1024, help="Max total sequence length.")
 
    # training
    parser.add_argument("--epochs", type=int,   default=1,    help="Number of DPO epochs.")
    parser.add_argument("--batch-size", type=int,   default=2,    help="Per device batch size.")
    parser.add_argument("--grad-accum", type=int,   default=4,    help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max-samples",type=int, default=None, help="Cap DPO training pairs. None = use all.")
 
    # logging
    parser.add_argument("--wandb-project", type=str, default="llm-align-dpo", help="Wandb project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
 
    # misc
    parser.add_argument("--dry-run", action="store_true", help="Print config without training.")
 
    return parser.parse_args()


# path
def build_paths(exp: str, dataset: str) -> tuple[Path, Path, Path]:
    """
    sft_checkpoint_dir: checkpoints/<exp>/sft/final/
    processed_dir: data/processed/<dataset>/
    dpo_checkpoint_dir: checkpoints/<exp>/dpo/
    """

    sft_checkpoint_dir = PROJECT_ROOT / "checkpoints" / exp / "sft" / "final"
    processed_dir      = PROJECT_ROOT / "data" / "processed" / dataset
    dpo_checkpoint_dir = PROJECT_ROOT / "checkpoints" / exp / "dpo"

    return sft_checkpoint_dir, processed_dir, dpo_checkpoint_dir


# main
def main() -> None:
    args = parse_args()
 
    sft_checkpoint_dir, processed_dir, dpo_checkpoint_dir = build_paths(
        args.exp, args.dataset
    )
 
    # validate SFT checkpoint exists
    if not sft_checkpoint_dir.exists():
        logger.error(
            f"SFT checkpoint not found: {sft_checkpoint_dir}\n"
            f"Run: python scripts/train_sft.py --exp {args.exp} --dataset {args.dataset} first."
        )
        sys.exit(1)
 
    # Validate DPO data exists
    if not (processed_dir / "dpo_train.jsonl").exists():
        logger.error(f"dpo_train.jsonl not found in {processed_dir}")
        sys.exit(1)
 

    cfg = DPOTrainingConfig(
        beta=args.beta,
        max_prompt_length=args.max_prompt_len,
        max_length=args.max_len,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        report_to="none" if args.no_wandb else "wandb",
    )
 
    exp_name = f"{args.exp}_dpo"
 
    # dry run
    if args.dry_run:
        print("\n--- DRY RUN ---\n")
        print(f"exp: {args.exp}")
        print(f"dataset: {args.dataset}")
        print(f"sft_checkpoint: {sft_checkpoint_dir}")
        print(f"processed_dir: {processed_dir}")
        print(f"dpo_checkpoint: {dpo_checkpoint_dir}")
        print(f"beta: {cfg.beta}")
        print(f"epochs: {cfg.num_train_epochs}")
        print(f"batch_size: {cfg.per_device_train_batch_size}")
        print(f"grad_accum: {cfg.gradient_accumulation_steps}")
        print(f"effective_batch: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
        print(f"learning_rate: {cfg.learning_rate}")
        print(f"max_prompt_length: {cfg.max_prompt_length}")
        print(f"max_length: {cfg.max_length}")
        print(f"max_samples: {args.max_samples or 'all'}")
        print(f"wandb logging: {'disabled' if args.no_wandb else 'enabled'}")
        print("\nNo training started. Remove --dry-run to execute.\n")
        return
 
    # log config
    logger.info(f"Experiment: {args.exp}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"SFT checkpoint: {sft_checkpoint_dir}")
    logger.info(f"Beta: {cfg.beta}")
    logger.info(f"Epochs: {cfg.num_train_epochs}")
    logger.info(f"Effective batch: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
 

    final_dir = run_dpo(
        sft_checkpoint_dir=sft_checkpoint_dir,
        processed_dir=processed_dir,
        checkpoint_dir=dpo_checkpoint_dir,
        exp_name=exp_name,
        cfg=cfg,
        wandb_project=args.wandb_project,
        max_samples=args.max_samples,
    )
 
    print("\n" + "=" * 50)
    print("DPO training done")
    print("=" * 50)
    print(f"Experiment: {args.exp}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {final_dir}")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()
    
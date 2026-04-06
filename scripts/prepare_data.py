import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# set root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.data.registry import get_dataset, list_datasets
 
# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# arg parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process a dataset for llm-align-dpo training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version label. Changing this creates a new processed folder, keeping old data intact.",
    )
    parser.add_argument(
        "--max-sft",
        type=int,
        default=None,
        help="Cap on SFT training rows. None = use all available.",
    )
    parser.add_argument(
        "--max-dpo",
        type=int,
        default=None,
        help="Cap on DPO training rows. None = use all available.",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=1.0,
        help="DPO pairs with (score_chosen - score_rejected) below this are dropped.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and splitting.",
    )
    parser.add_argument(
        "--sft-eval-frac",
        type=float,
        default=0.05,
        help="Fraction of SFT data held out for eval.",
    )
    parser.add_argument(
        "--dpo-eval-frac",
        type=float,
        default=0.05,
        help="Fraction of DPO data held out for eval.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without downloading or writing anything.",
    )
 
    return parser.parse_args()


# path setup
def build_paths(dataset_name: str, version: str) -> tuple[Path, Path]:
    """
    Returns (raw_dir, processed_dir) for a given dataset + version.
 
    raw_dir path(shared): data/raw/<dataset_name>/
    processed_dir path (versioned): data/processed/<dataset_name>_<version>/
    """
    raw_dir       = PROJECT_ROOT / "data" / "raw"       / dataset_name
    processed_dir = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_{version}"
    return raw_dir, processed_dir


# huggingface token check
def check_hf_token() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN not found in environment. "
            "Public datasets will still work, but gated models will fail. "
            "Add HF_TOKEN to your .env file to avoid this."
        )
    else:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logger.info("HF_TOKEN loaded.")


def main() -> None:
    args = parse_args()
 
    raw_dir, processed_dir = build_paths(args.dataset, args.version)
 
    # dry run 
    if args.dry_run:
        print("\n--- DRY RUN ---\n")
        print(f"dataset: {args.dataset}")
        print(f"version: {args.version}")
        print(f"raw_dir: {raw_dir}")
        print(f"processed_dir: {processed_dir}")
        print(f"max_sft: {args.max_sft or 'all'}")
        print(f"max_dpo: {args.max_dpo or 'all'}")
        print(f"min_score_gap: {args.min_score_gap}")
        print(f"seed: {args.seed}")
        print(f"sft_eval_frac: {args.sft_eval_frac}")
        print(f"dpo_eval_frac: {args.dpo_eval_frac}")
        print("\nNo files written. Remove --dry-run to execute.\n")
        return
 
    
    check_hf_token()
 
    # log
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Raw dir: {raw_dir}")
    logger.info(f"Processed dir: {processed_dir}")
    logger.info(f"Max SFT rows: {args.max_sft or 'all'}")
    logger.info(f"Max DPO rows: {args.max_dpo or 'all'}")
    logger.info(f"Min score gap: {args.min_score_gap}")
    logger.info(f"Seed: {args.seed}")
 
    # warn if processed_dir already exists — won't overwrite, just re-runs
    if processed_dir.exists():
        logger.warning(
            f"{processed_dir} already exists. "
            "Files will be overwritten. Use --version v2 to keep old data."
        )
 
    ds = get_dataset(
        name=args.dataset,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        random_seed=args.seed,
        min_score_gap=args.min_score_gap,
        sft_eval_frac=args.sft_eval_frac,
        dpo_eval_frac=args.dpo_eval_frac,
    )
 
    manifest = ds.prepare(
        max_sft=args.max_sft,
        max_dpo=args.max_dpo,
    )
 
    # data prep summary
    print("\n" + "=" * 50)
    print(f"data prep done!\n{'-' * 25}\nSummary")
    print(f"Dataset: {manifest.dataset_name} ({manifest.version})")
    print(f"SFT train: {manifest.sft_train_rows:,} rows")
    print(f"SFT eval: {manifest.sft_eval_rows:,} rows")
    print(f"DPO train: {manifest.dpo_train_rows:,} rows")
    print(f"DPO eval: {manifest.dpo_eval_rows:,} rows")
    print(f"Min score gap: {manifest.min_score_gap}")
    print(f"Seed: {manifest.random_seed}")
    print(f"Manifest: {processed_dir / 'manifest.json'}")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
from typing import Type
from src.data.base_dataset import BaseDataset

from src.data.dataset_ultrafeedback import UltraFeedbackDataset
from src.data.dataset_openhermes import OpenHermesDataset


# dataset keys
_REGISTRY: dict[str, Type[BaseDataset]] = {
    "ultrafeedback": UltraFeedbackDataset,
    "openhermes": OpenHermesDataset
}


# list datasets
def list_datasets() -> list[str]:
    """
    return all regustered dataset names
    """
    return sorted(_REGISTRY.keys())

# get dataset class
def get_dataset_class(name: str) -> Type[BaseDataset]:
    """
    Return the class for a dataset without instantiating it 
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        available = ", ".join(list_datasets())
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {available}\n"
            f"To add a new dataset, see the docstring in registry.py."
        )
    return _REGISTRY[name]


# get dataset
def get_dataset(
        name: str,
        raw_dir: Path,
        processed_dir: Path,
        random_seed: int = 42,
        min_score_gap: float = 1.0,
        sft_eval_frac: float = 0.05,
        dpo_eval_frac: float = 0.05
) -> BaseDataset:
    """
    Instantiate and return dataset by name
    """
    cls = get_dataset_class(name)
    return cls(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        random_seed=random_seed,
        min_score_gap=min_score_gap,
        sft_eval_frac=sft_eval_frac,
        dpo_eval_frac=dpo_eval_frac,
    )


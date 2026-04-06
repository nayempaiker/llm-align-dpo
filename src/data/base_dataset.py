from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class SFTSample:
    """
    One supervised fine-tuning example
    """
    instruction: str
    output: str
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SFTSample":
        return cls(
            instruction=d["instruction"],
            output=d["output"],
            source=d.get("source", ""),
            metadata=d.get("metadata", {}),
        )

    def to_chatml(self) -> str:
        """
        CHatml format string
        """
        return (
            f"<|im_start|>user\n{self.instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.output}<|im_end|>"
        )


@dataclass
class DPOSample:
    """
    One preference pair for DPO training.
    """
    prompt: str
    chosen: str
    rejected: str
    score_chosen: float = 0.0 
    score_rejected: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def score_gap(self) -> float:
        """
        Gap to filter weak pairs
        """
        return self.score_chosen - self.score_rejected

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DPOSample":
        return cls(
            prompt=d["prompt"],
            chosen=d["chosen"],
            rejected=d["rejected"],
            score_chosen=d.get("score_chosen", 0.0),   
            score_rejected=d.get("score_rejected", 0.0),
            source=d.get("source", ""),
            metadata=d.get("metadata", {}),
        )

    def to_chatml(self) -> dict:
        """
        Chatml format dict
        """
        prompt_str = f"<|im_start|>user\n{self.prompt}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt":   prompt_str,
            "chosen":   self.chosen + "<|im_end|>",
            "rejected": self.rejected + "<|im_end|>",
        }


@dataclass
class DatasetManifest:
    """
    Manifest for data preparation
    """
    dataset_name: str
    version: str
    source_files: list[str]
    sft_train_rows: int
    sft_eval_rows: int
    dpo_train_rows: int
    dpo_eval_rows: int
    min_score_gap: float
    random_seed: int
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Manifest saved -> {path}")

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        with open(path) as f:
            return cls(**json.load(f))


class BaseDataset(ABC):
    """
    ABstract base — every dataset adapter inherits from this class
    """

    NAME: str = ""
    VERSION: str = "v1"

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        random_seed: int = 42,
        min_score_gap: float = 1.0,
        sft_eval_frac: float = 0.05,
        dpo_eval_frac: float = 0.05,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.random_seed = random_seed
        self.min_score_gap = min_score_gap
        self.sft_eval_frac = sft_eval_frac
        self.dpo_eval_frac = dpo_eval_frac
        self.processed_dir.mkdir(parents=True, exist_ok=True)



    # abstract 

    @abstractmethod                         
    def download(self) -> None:
        """
        Download raw data into self.raw_dir
        """
        ...

    @abstractmethod
    def iter_sft(self) -> Iterator[SFTSample]:
        """
        Yield SFTSample objects from raw data, one at a time
        """
        ...

    @abstractmethod
    def iter_dpo(self) -> Iterator[DPOSample]:
        """
        Yield DPOSample objects from raw data, one at a time
        """
        ...


    def prepare(self, max_sft: int | None = None, max_dpo: int | None = None) -> DatasetManifest:
        """
        Full pipeline: download → parse → filter → split → save
        """
        random.seed(self.random_seed)

        logger.info(f"[{self.NAME}] Downloading raw data...")   # FIX: ionfo typo
        self.download()

        # sft
        logger.info(f"[{self.NAME}] Building SFT split...")
        sft_samples = list(self.iter_sft())
        random.shuffle(sft_samples)

        if max_sft:
            sft_samples = sft_samples[:max_sft]

        n_sft_eval = max(1, int(len(sft_samples) * self.sft_eval_frac))
        sft_eval   = sft_samples[:n_sft_eval]
        sft_train  = sft_samples[n_sft_eval:]

        self._save_jsonl(sft_train, self.processed_dir / "sft_train.jsonl", lambda s: s.to_dict())
        self._save_jsonl(sft_eval,  self.processed_dir / "sft_eval.jsonl",  lambda s: s.to_dict())



        # dpo
        logger.info(f"[{self.NAME}] Building DPO split (min_score_gap={self.min_score_gap})...")
        dpo_samples = [s for s in self.iter_dpo() if s.score_gap >= self.min_score_gap]
        logger.info(f"[{self.NAME}] DPO pairs after gap filter: {len(dpo_samples)}")

        random.shuffle(dpo_samples)

        if max_dpo:
            dpo_samples = dpo_samples[:max_dpo]

        n_dpo_eval = max(1, int(len(dpo_samples) * self.dpo_eval_frac))
        dpo_eval   = dpo_samples[:n_dpo_eval]
        dpo_train  = dpo_samples[n_dpo_eval:]

        self._save_jsonl(dpo_train, self.processed_dir / "dpo_train.jsonl", lambda s: s.to_dict())
        self._save_jsonl(dpo_eval,  self.processed_dir / "dpo_eval.jsonl",  lambda s: s.to_dict())


        # manifest
        manifest = DatasetManifest(
            dataset_name=self.NAME,
            version=self.VERSION,
            source_files=[str(p) for p in sorted(self.raw_dir.glob("**/*")) if p.is_file()],
            sft_train_rows=len(sft_train),
            sft_eval_rows=len(sft_eval),
            dpo_train_rows=len(dpo_train),
            dpo_eval_rows=len(dpo_eval),
            min_score_gap=self.min_score_gap,
            random_seed=self.random_seed,
        )
        manifest.save(self.processed_dir / "manifest.json")

        logger.info(
            f"[{self.NAME}] Done. "
            f"SFT: {len(sft_train)} train / {len(sft_eval)} eval | "
            f"DPO: {len(dpo_train)} train / {len(dpo_eval)} eval"
        )
        return manifest



    def load_sft_train(self) -> list[SFTSample]:
        return self._load_jsonl(self.processed_dir / "sft_train.jsonl", SFTSample.from_dict)


    def load_sft_eval(self) -> list[SFTSample]:
        return self._load_jsonl(self.processed_dir / "sft_eval.jsonl", SFTSample.from_dict)


    def load_dpo_train(self) -> list[DPOSample]:
        return self._load_jsonl(self.processed_dir / "dpo_train.jsonl", DPOSample.from_dict)


    def load_dpo_eval(self) -> list[DPOSample]:
        return self._load_jsonl(self.processed_dir / "dpo_eval.jsonl", DPOSample.from_dict)


    @staticmethod
    def _save_jsonl(items: list, path: Path, serializer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(serializer(item)) + "\n")
        logger.info(f"Saved {len(items):,} rows -> {path}")



    @staticmethod
    def _load_jsonl(path: Path, deserializer) -> list:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_data.py --dataset <name> first."
            )
        with open(path) as f:
            return [deserializer(json.loads(line)) for line in f if line.strip()]
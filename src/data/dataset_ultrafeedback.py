from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from datasets import load_dataset, load_from_disk

from src.data.base_dataset import BaseDataset, SFTSample, DPOSample

logger = logging.getLogger(__name__)

_RATING_KEYS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]

class UltraFeedbackDataset(BaseDataset):
    NAME = "ultrafeedback"
    VERSION = "v1"

    def download(self) -> None:
        """
        Download and process ultrafeedback dataset from huggingface
        """
        marker = self.raw_dir / ".downloaded"

        if marker.exists():
            logger.info("[ultrafeedback] Raw data already present, skipping download")
            return

        logger.info("[ultrafeedback] Downloading from HuggingFace Hub...")

        # get dataset from huggingface: openbmb
        ds = load_dataset("openbmb/UltraFeedback", split="train")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(self.raw_dir))

        marker.touch()
        logger.info(f"[ultrafeedback] Downloaded {len(ds):,} rows → {self.raw_dir}")

    # iteration sft
    def iter_sft(self) -> Iterator[SFTSample]:
        """
        Iterate through the data, pick the completion with SFTSample preference
        """
        for row in self._iter_raw():
            instruction = row.get("instruction", "").strip()

            if not instruction:
                continue
 
            best = self._best_completion(row)

            if best is None:
                continue
 
            yield SFTSample(
                instruction=instruction,
                output=best["response"].strip(),
                source=self.NAME,
                metadata={"model": best.get("model", "")},
            )


    # iteration dpo
    def iter_dpo(self) -> Iterator[DPOSample]:
        """
        Iterate through the data yield datapoint with DPOSample preference
        """
        for row in self._iter_raw():
            instruction = row.get("instruction", "").strip()
            completions = row.get("completions", [])
 
            if not instruction or len(completions) < 2:
                continue
 
            scored = self._score_completions(completions)
            if len(scored) < 2:
                continue
 
            # highest score = chosen, lowest score = rejected
            chosen_entry   = scored[-1]
            rejected_entry = scored[0]
 
            yield DPOSample(
                prompt=instruction,
                chosen=chosen_entry["response"].strip(),
                rejected=rejected_entry["response"].strip(),
                score_chosen=chosen_entry["score"],
                score_rejected=rejected_entry["score"],
                source=self.NAME,
                metadata={
                    "chosen_model":   chosen_entry.get("model", ""),
                    "rejected_model": rejected_entry.get("model", ""),
                },
            )

    # iteration raw
    def _iter_raw(self):
        """
        Load from disk and yield raw HuggingFace rows
        """
        
        ds = load_from_disk(str(self.raw_dir))
        for row in ds:
            yield row

    # score completations
    def _score_completions(self, completions: list[dict]) -> list[dict]:
        """
        Iterate through the completions, attach a scalar score to each completion and sort ascending
        """
        scored = []
        for c in completions:
            score = self._extract_score(c)
            if score is not None:
                scored.append({**c, "score": score})
        return sorted(scored, key=lambda x: x["score"])


    @staticmethod
    def _extract_score(completion: dict) -> float | None:
        """
        Average the four rating dimensions into one scalar score. Falls back to completion["overall_score"] if annotations missing.
        """
        try:
            annotations = completion.get("annotations", {})
            if annotations:
                scores = []
                for key in _RATING_KEYS:
                    val = annotations.get(key, {})
                    if isinstance(val, dict) and "Rating" in val:
                        scores.append(float(val["Rating"]))
                if scores:
                    return sum(scores) / len(scores)
 
            # fallback: overall_score stored directly on the completion
            overall = completion.get("overall_score")
            if overall is not None:
                return float(overall)
 
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    # best completion
    def _best_completion(self, row: dict) -> dict | None:
        completions = row.get("completions", [])
        scored = self._score_completions(completions)
        return scored[-1] if scored else None
        
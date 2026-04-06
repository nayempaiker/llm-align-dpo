from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator
from datasets import load_dataset, load_from_disk

from src.data.base_dataset import BaseDataset, SFTSample, DPOSample

logger = logging.getLogger(__name__)

_SOURCE_WEIGHTS: dict[str, float] = {
    "gpt4_alpaca":            1.0,
    "cot_alpaca_gpt4":        1.0,
    "airoboros2.2":           0.8,
    "joke_explanation":       0.3,   
    "metamath":               0.5,
    "platypus":               0.7,
    "slimorca":               0.6,
    "unknown":                0.4,
}
_DEFAULT_WEIGHT = 0.5 


class OpenHermesDataset(BaseDataset):
    NAME    = "openhermes"
    VERSION = "v1"

    def download(self) -> None:
        """
        Download and process Openher,mes dataset from huggingface
        """

        marker = self.raw_dir / ".downloaded"
        if marker.exists():
            logger.info("[openhermes] Raw data already present, skipping download.")
            return
        
        logger.info("[openhermes] Downloading from HuggingFace Hub...")

        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=False)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(self.raw_dir))
        marker.touch()
        logger.info(f"[openhermes] Downloaded → {self.raw_dir}")

    
    # iterate sft
    def iter_sft(self) -> Iterator[SFTSample]:
        """
        Iterate through the data, apply probabilistic sampling and skip short output,
        yield with SFTSample preference
        """
        for row in self._iter_raw():
            source = row.get("source", "unknown")
            weight = _SOURCE_WEIGHTS.get(source, _DEFAULT_WEIGHT)
            
            # probabilistic keep — fast, no pre-filtering needed
            if random.random() > weight:
                continue
        
            instruction, output = self._extract_text(row)
            if not instruction or not output:
                continue
                
            # skip short output
            if len(output.split()) < 10:
                continue

            yield SFTSample(
                instruction=instruction,
                output = output,
                source = f"{self.NAME}/{source}",
                metadata = {"source_tag": source}
            )

    # iterate dpo
    def iter_dpo(self) -> Iterator[DPOSample]:
        """
        Openhermes is SFT only dataset
        """
        logger.info(
            "[openhermes] iter_dpo called but OpenHermes has no preference pairs. "
            "Use UltraFeedback for DPO data."
        )

        return iter([])

    # iterate raw
    def _iter_raw(self):
        ds = load_from_disk(str(self.raw_dir))

        for row in ds:
            yield row


    # extract text
    @staticmethod
    def _extract_text(row: dict) -> tuple[str, str]:
        """
        From multi-turn conversations, extract human turn as instruction and gpt turn as output
        """
        conversations = row.get("conversations", [])
        instruction, output = "", ""

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "").strip()

            if role == "human" and not instruction:
                instruction = value
            elif role == "gpt" and not output:
                output = value

        
        return instruction, output

    

    

        

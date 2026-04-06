"""
test_data_layer.py
------------------
Run this before touching any real data.
No downloads, no GPU, no external calls.

Usage:
    python test_data_layer.py
"""

import sys
import json
import tempfile
from pathlib import Path

from src.data.base_dataset import SFTSample, DPOSample, DatasetManifest, BaseDataset
from src.data.registry import get_dataset, get_dataset_class, list_datasets


# SFTSample test

print("\nTesting SFTSample serialization...")
sft = SFTSample(
    instruction="What is gradient descent?",
    output="It is an optimization algorithm that minimizes a loss function.",
    source="test",
)
d = sft.to_dict()

sft2 = SFTSample.from_dict(d)
assert sft.instruction == sft2.instruction, "instruction mismatch"
assert sft.output == sft2.output, "output mismatch"
chatml = sft.to_chatml()

assert "<|im_start|>user" in chatml
assert "<|im_start|>assistant" in chatml
print("OK — to_dict / from_dict / to_chatml all pass")



# DPOSample test

print("\nTesting DPOSample serialization...")
dpo = DPOSample(
    prompt="Explain recursion.",
    chosen="Recursion is when a function calls itself with a smaller input.",
    rejected="Recursion is recursion.",
    score_chosen=8.5,
    score_rejected=3.0,
    source="test",
)
assert dpo.score_gap == 5.5, f"score_gap wrong: {dpo.score_gap}"
d = dpo.to_dict()

dpo2 = DPOSample.from_dict(d)
assert dpo2.score_chosen == 8.5, "score_chosen lost in round-trip"
assert dpo2.score_rejected == 3.0, "score_rejected lost in round-trip"
chatml = dpo.to_chatml()

assert "prompt" in chatml and "chosen" in chatml and "rejected" in chatml
print("OK — score_gap, round-trip, to_chatml all pass")



# Registry 
print("\nTesting registry...")
datasets = list_datasets()
assert "ultrafeedback" in datasets, "ultrafeedback missing from registry"
assert "openhermes" in datasets, "openhermes missing from registry"
print(f"OK — registered datasets: {datasets}")

try:
    get_dataset_class("does_not_exist")
    print("   FAIL — should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"OK — unknown dataset raises ValueError correctly")



# BaseDataset._save_jsonl / _load_jsonl
print("\nTesting _save_jsonl / _load_jsonl...")
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "test.jsonl"
    samples = [
        SFTSample("q1", "a1", "test"),
        SFTSample("q2", "a2", "test"),
    ]
    
    BaseDataset._save_jsonl(samples, path, lambda s: s.to_dict())
    loaded = BaseDataset._load_jsonl(path, SFTSample.from_dict)

    assert len(loaded) == 2, f"expected 2 rows, got {len(loaded)}"
    assert loaded[0].instruction == "q1"
    assert loaded[1].instruction == "q2"
print("OK — save and load round-trip with 2 rows")



# DatasetManifest save/load
print("\nTesting DatasetManifest...")
with tempfile.TemporaryDirectory() as tmpdir:
    manifest = DatasetManifest(
        dataset_name="test",
        version="v1",
        source_files=[],
        sft_train_rows=1000,
        sft_eval_rows=50,
        dpo_train_rows=800,
        dpo_eval_rows=40,
        min_score_gap=1.0,
        random_seed=42,
    )

    path = Path(tmpdir) / "manifest.json"
    manifest.save(path)
    loaded = DatasetManifest.load(path)

    assert loaded.sft_train_rows == 1000
    assert loaded.min_score_gap == 1.0
print("OK — manifest save/load pass")


print("\n" + "="*40)
print("All tests passed. Data layer is working correctly.")
print("="*40)
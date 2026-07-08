from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def load_sft_dataset(path_or_name: str, split: str | None = None) -> Dataset | DatasetDict:
    """Load a dataset either from the HF Hub or from local JSONL file(s).

    Local paths ending in .jsonl/.json are loaded via the `json` builder;
    anything else is treated as a Hub dataset id.
    """
    p = Path(path_or_name)
    if p.suffix in {".jsonl", ".json"}:
        ds = load_dataset("json", data_files=str(p), split=split or "train")
    else:
        ds = load_dataset(path_or_name, split=split)
    return ds


def train_eval_split(dataset: Dataset, eval_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    if not 0 < eval_ratio < 1:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return DatasetDict(train=split["train"], eval=split["test"])

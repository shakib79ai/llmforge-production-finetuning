from __future__ import annotations

from datasets import Dataset

REQUIRED_ROLES = {"user", "assistant"}


class DatasetValidationError(ValueError):
    pass


def validate_sft_dataset(dataset: Dataset) -> None:
    """Raise DatasetValidationError if the dataset doesn't match the expected
    {"messages": [{"role": ..., "content": ...}, ...]} schema."""
    if "messages" not in dataset.column_names:
        raise DatasetValidationError(
            f"Expected a 'messages' column, got {dataset.column_names}"
        )

    sample_size = min(len(dataset), 200)
    for i in range(sample_size):
        messages = dataset[i]["messages"]
        if not isinstance(messages, list) or not messages:
            raise DatasetValidationError(f"Row {i}: 'messages' must be a non-empty list")

        roles = {m.get("role") for m in messages}
        if not roles & REQUIRED_ROLES:
            raise DatasetValidationError(
                f"Row {i}: expected at least one of roles {REQUIRED_ROLES}, got {roles}"
            )
        for m in messages:
            if "role" not in m or "content" not in m:
                raise DatasetValidationError(f"Row {i}: message missing 'role' or 'content': {m}")
            if not isinstance(m["content"], str) or not m["content"].strip():
                raise DatasetValidationError(f"Row {i}: empty content for role {m.get('role')}")

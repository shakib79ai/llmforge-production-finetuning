import pytest
from datasets import Dataset

from llmforge.data.validation import DatasetValidationError, validate_sft_dataset


def test_validate_sft_dataset_accepts_well_formed_rows():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is LoRA?"},
                    {"role": "assistant", "content": "A parameter-efficient fine-tuning method."},
                ]
            }
        ]
    )
    validate_sft_dataset(dataset)  # should not raise


def test_validate_sft_dataset_rejects_missing_column():
    dataset = Dataset.from_list([{"text": "no messages column"}])
    with pytest.raises(DatasetValidationError):
        validate_sft_dataset(dataset)


def test_validate_sft_dataset_rejects_empty_content():
    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": ""}]}]
    )
    with pytest.raises(DatasetValidationError):
        validate_sft_dataset(dataset)

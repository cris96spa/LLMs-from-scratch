import pytest
import torch

from LLMs_from_scratch.configs.models import SpamDatasetConfig, TokenizerConfig
from LLMs_from_scratch.dataloader.spam_dataset import SpamDataset
from LLMs_from_scratch.tokenizer import Tokenizer


@pytest.fixture
def spam_dataset_config() -> SpamDatasetConfig:
    return SpamDatasetConfig(
        file_path="data/spam_dataset/SMSSpamCollection.tsv",
        max_length=None,
        pad_token_id=50256,
    )


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        encoding=None,
        model_name="gpt2",
        allowed_special={"<|endoftext|>"},
    )


@pytest.fixture
def spam_dataset(
    spam_dataset_config: SpamDatasetConfig, tokenizer_config: TokenizerConfig
) -> SpamDataset:
    tokenizer = Tokenizer(config=tokenizer_config)
    return SpamDataset(config=spam_dataset_config, tokenizer=tokenizer)


class TestSpamDataset:
    def test_spam_dataset_length(self, spam_dataset: SpamDataset):
        assert len(spam_dataset) > 0, "The dataset should not be empty."

    def test_spam_dataset_item(self, spam_dataset: SpamDataset):
        item = spam_dataset[0]
        assert isinstance(item, tuple), (
            "Each item should be a tuple of (encoded_text, label)."
        )
        assert len(item) == 2, (
            "Each item should contain two elements: (encoded_text, label)."
        )
        encoded_text, label = item
        assert isinstance(encoded_text, torch.Tensor), (
            "Encoded text should be a torch.Tensor."
        )
        assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor."

    def test_spam_dataset_encoding(self, spam_dataset: SpamDataset):
        encoded_text, _ = spam_dataset[0]
        assert encoded_text.dtype == torch.int32, (
            "Encoded text should be of type torch.int32."
        )
        assert encoded_text.ndim == 1, "Encoded text should be a 1D tensor."
        assert encoded_text.shape[0] <= spam_dataset._max_length, (
            "Encoded text should not exceed the maximum length."
        )

        # Ensure that the decoded text matches the original text in the dataset
        encoded_tokens = encoded_text.tolist()
        try:
            first_padding_index = encoded_tokens.index(spam_dataset._pad_token_id)
        except ValueError:
            first_padding_index = None

        assert (
            spam_dataset._tokenizer.decode(encoded_tokens[:first_padding_index])
            == spam_dataset._data[spam_dataset._text_column_name][0]
        )

    def test_spam_dataset_label_mapping(self, spam_dataset: SpamDataset):
        _, label = spam_dataset[0]
        assert label.dtype == torch.int32, "Label should be of type torch.int32."
        assert label.ndim == 0, "Label should be a scalar tensor."
        assert label.item() in spam_dataset.label_mapping.values(), (
            "Label value should be one of the mapped integer IDs."
        )
        # Ensure that the label mapping is correct
        original_label_id = spam_dataset._data[spam_dataset._label_column_name][0]

        assert label.item() == original_label_id, (
            "The label ID should match the expected ID from the label mapping."
        )

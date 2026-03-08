import polars as pl
import torch
from tiktoken import Encoding
from torch.utils.data import Dataset

from LLMs_from_scratch.configs.models import SpamDatasetConfig
from utils.logger import logger

SPAM_DATASET_ORIGINAL_NAME = "SMSSpamCollection"

logger = logger.bind(app="Dataset Utils")


class SpamDataset(Dataset):
    def __init__(self, config: SpamDatasetConfig, tokenizer: Encoding) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._logger = logger.bind(app="SpamDataset")
        self._pad_token_id = config.pad_token_id

        self._logger.info(f"Loading dataset from {config.file_path}...")
        self._data = pl.read_csv(
            config.file_path,
            separator="\t",
            has_header=False,
            new_columns=config.column_names,
            quote_char=None,
        )

        # Map label names to integer IDs using the label_mapping from the config
        self._data = self._data.with_columns(
            pl.col(config.label_column_name)
            .replace(config.label_mapping)
            .cast(pl.Int32)
        )

        self._logger.info(f"Dataset loaded with {self._data.shape[0]} examples.")

        self._label_column_name = config.label_column_name
        self._text_column_name = config.text_column_name

        self._logger.info(f"Encoding the dataset[{self._text_column_name}]...")
        self._encoded_texts = [
            tokenizer.encode(text) for text in self._data[self._text_column_name]
        ]

        self._max_length = config.max_length
        if self._max_length is None:
            self._max_length = self._longest_encoded_text_length()
        else:
            self._truncate_encoded_texts()

        self._pad_encoded_texts()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encoded_texts[index]
        label = self._data[self._label_column_name][index]
        return torch.tensor(encoded, dtype=torch.int32), torch.tensor(
            label, dtype=torch.int32
        )

    def __len__(self) -> int:
        return len(self._encoded_texts)

    def _longest_encoded_text_length(self) -> int:
        """Calculate the length of the longest encoded text in the dataset.

        Returns:
            int: The length of the longest encoded text.
        """
        return max(len(encoded) for encoded in self._encoded_texts)

    def _truncate_encoded_texts(self) -> None:
        """Truncate the encoded texts to the maximum length specified in the config."""
        logger.info(f"Truncating encoded texts to max_length: {self._max_length}...")
        self._encoded_texts = [
            encoded[: self._max_length] for encoded in self._encoded_texts
        ]

    def _pad_encoded_texts(self) -> None:
        """Pad the encoded texts to the maximum length specified in the config."""
        logger.info(
            f"Padding encoded texts to max_length: {self._max_length} with pad_token_id: {self._config.pad_token_id}..."
        )
        self._encoded_texts = [
            encoded + [self._pad_token_id] * (self._max_length - len(encoded))
            for encoded in self._encoded_texts
        ]

    @property
    def label_mapping(self) -> dict[str, int]:
        """Get the label mapping from the config."""
        return self._config.label_mapping

    def get_label_name(self, label_id: int) -> str:
        """Get the label name corresponding to a given label ID.

        Args:
            label_id (int): The label ID to look up.

        Returns:
            str: The corresponding label
            name, or "Unknown" if the label ID is not found in the mapping.
        """
        for name, id in self._config.label_mapping.items():
            if id == label_id:
                return name
        raise ValueError(f"Label ID {label_id} not found in label mapping.")

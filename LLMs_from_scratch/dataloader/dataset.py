import numpy as np
import torch
from torch.utils.data import Dataset

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import GPTDatasetConfig
from LLMs_from_scratch.tokenizer import Tokenizer


class GPTDataset(Dataset):
    def __init__(self, config: GPTDatasetConfig):
        self._config = config
        self._max_length = self._config.max_length
        self._stride = self._config.stride

        self.tokenizer: Tokenizer = Tokenizer(ConfigsProvider().tokenizer_config)
        self.raw_text = self._config.file_path.read_text(encoding="utf-8")
        self._load_data()

    def _load_data(self) -> None:
        ids = np.array(self.tokenizer.encode(self.raw_text), dtype=np.float32)

        if ids.size < self._max_length:
            raise ValueError("Input text is shorter than the maximum length.")

        inputs = np.lib.stride_tricks.sliding_window_view(
            ids, window_shape=self._max_length
        )[:: self._stride]
        targets = np.lib.stride_tricks.sliding_window_view(
            ids[1:], window_shape=self._max_length
        )[:: self._stride]

        self.input_ids = torch.tensor(inputs, dtype=torch.int32)
        self.target_ids = torch.tensor(targets, dtype=torch.int32)

    def _load_data_tensor(self) -> None:
        ids = torch.tensor(self.tokenizer.encode(self.raw_text), dtype=torch.int32)

        if ids.numel() < self._max_length:
            raise ValueError("Input text is shorter than the maximum length.")

        inputs = ids.unfold(0, self._max_length, self._stride)
        targets = ids[1:].unfold(0, self._max_length, self._stride)

        self.input_ids = inputs
        self.target_ids = targets

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        return self.input_ids[idx], self.target_ids[idx]

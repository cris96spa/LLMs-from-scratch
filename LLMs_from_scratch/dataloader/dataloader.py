from torch.utils.data import DataLoader

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import DataLoaderConfig
from LLMs_from_scratch.dataloader.dataset import GPTDataset
from LLMs_from_scratch.tokenizer import Tokenizer


def create_dataloader(config: DataLoaderConfig, dataset: GPTDataset) -> DataLoader:
    """Create a DataLoader instance based on the provided configuration."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )


if __name__ == "__main__":
    dataloader_config = ConfigsProvider().dataloader_config
    tokenizer_config = ConfigsProvider().tokenizer_config
    tokenizer = Tokenizer(tokenizer_config)

    gpt_dataset_config = ConfigsProvider().gpt_dataset_config

    dataloader = create_dataloader(
        dataloader_config,
        GPTDataset(gpt_dataset_config, tokenizer),
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    pass

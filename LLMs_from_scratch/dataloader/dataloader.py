from torch.utils.data import DataLoader

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import DataLoaderConfig
from LLMs_from_scratch.dataloader.dataset import GPTDataset


def create_dataloader(config: DataLoaderConfig) -> DataLoader:
    """Create a DataLoader instance based on the provided configuration."""
    dataset = GPTDataset(ConfigsProvider().gpt_dataset_config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )


if __name__ == "__main__":
    dataloader = create_dataloader(ConfigsProvider().dataloader_config)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    pass

import zipfile
from pathlib import Path

import polars as pl
import requests

from LLMs_from_scratch.configs.models import TrainValTestSplit
from utils.configs_provider import BaseConfigProvider
from utils.logger import logger

logger = logger.bind(app="Dataset Utils")

SPAM_DATASET_ORIGINAL_NAME = "SMSSpamCollection"


def download_and_unzip_spam_data(
    url: str, zip_path: Path, extracted_path: Path, data_file: Path
):
    """Download and unzip the spam data if it doesn't already exist.

    Args:
        url (str): The URL to download the zip file from.
        zip_path (Path): The local path to save the downloaded zip file.
        extracted_path (Path): The local path to extract the contents of the zip file.
        data_file (Path): The expected path of the extracted data file after renaming.
    """

    if data_file.exists():
        logger.info(f"{data_file} already exists. Skipping download.")
        return

    logger.info(f"Downloading {url} to {zip_path}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"Unzipping the downloaded file to {extracted_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    logger.info("Adding .tsv file extension to the extracted file...")
    original_file = extracted_path / SPAM_DATASET_ORIGINAL_NAME
    if original_file.exists():
        original_file.rename(data_file)
        logger.info(f"Renamed {original_file} to {data_file}.")
    else:
        logger.error(f"Expected file {original_file} not found after extraction.")
        raise FileNotFoundError(f"{original_file} not found.")

    logger.info("Cleaning up the downloaded zip file...")
    zip_path.unlink()
    logger.info("Download and extraction complete.")


def create_balanced_dataset(data: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """Create a balanced dataset by sampling an equal number of examples from each class.

    Args:
        data (pl.DataFrame): The input DataFrame containing the data.
        label_col (str): The name of the column containing the labels.

    Returns:
        pl.DataFrame: A balanced DataFrame with an equal number of examples from each class.
    """
    logger.info("Creating a balanced dataset...")

    unique_classes = data[label_col].unique()
    logger.info(f"Unique classes found: {unique_classes}")

    min_class_count = min(
        data.filter(pl.col(label_col) == class_label).shape[0]
        for class_label in unique_classes
    )
    data_balanced = pl.concat(
        [
            data.filter(pl.col(label_col) == class_label).sample(n=min_class_count)
            for class_label in unique_classes
        ]
    )
    logger.info(
        f"Balanced dataset created with {min_class_count} examples per class, total {data_balanced.shape[0]} examples."
    )
    return data_balanced


def train_val_test_split(
    data: pl.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1
) -> TrainValTestSplit:
    """Split the dataset into training, validation, and test sets.

    The test set fraction is implicitly defined as 1 - (train_frac + val_frac).
    Args:
        data (pl.DataFrame): The input DataFrame containing the data.
        train_frac (float): The fraction of the data to use for training.
        val_frac (float): The fraction of the data to use for validation.

    Returns:
        TrainValTestSplit: A TrainValTestSplit object containing the train, val, and
        test DataFrames.
    """
    seed = BaseConfigProvider().seed
    logger.info(
        f"Splitting the dataset into train, val, and test sets with seed {seed}..."
    )
    data_shuffled = data.sample(
        fraction=1.0, seed=seed, shuffle=True, with_replacement=False
    )
    train_idx = int(train_frac * data_shuffled.shape[0])
    val_idx = int((train_frac + val_frac) * data_shuffled.shape[0])
    train_data = data_shuffled[:train_idx]
    val_data = data_shuffled[train_idx:val_idx]
    test_data = data_shuffled[val_idx:]
    logger.info(
        f"Dataset split complete: Train set: {train_data.shape[0]} examples, "
        f"Val set: {val_data.shape[0]} examples, Test set: {test_data.shape[0]} examples."
    )
    return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

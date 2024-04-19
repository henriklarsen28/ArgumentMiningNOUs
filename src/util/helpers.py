import json

import pandas as pd
import torch
from datasets import Dataset


<<<<<<< HEAD
def id2label(idx):
=======
def id2label(idx):
    # Read json file
    with open('../../dataset/label2id.json') as f:
        dicti = json.loads(f.read())
        id2lab = {idx: label for label, idx in dicti.items()}
        label = id2lab[idx]
        return label
    
def labels():
    with open('../../dataset/label2id.json') as f:
        # Convert json file to dictionary
        dicti = json.loads(f.read())
        return dicti.keys()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_dataset_from_csv(csv_path, test_size=0.1) -> (Dataset, int):
    """
    Loads a dataset from a CSV file, preprocesses it, and splits it into training and test sets.

    Args:
        self (str): The file path to the CSV file containing the dataset.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' datasets.
    """
    # Load data from CSV
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']].dropna()
    full_dataset = Dataset.from_pandas(df)

    train_test_split = full_dataset.train_test_split(seed=42, test_size=test_size)
    num_labels = len(set(df['label']))
    return train_test_split, num_labels

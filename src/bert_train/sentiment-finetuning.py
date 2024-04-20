from finetuning import train_and_save_classifiers
from datasets import load_dataset
import pandas as pd

path = "../../dataset/norec.csv"


def extract_dataset():
    dataset = load_dataset('ltg/norec', split='train')
    df = dataset.to_pandas()

    df['rating'] = df['rating'] - 1  # reduce to index
    df = df.rename(columns={'rating': 'label'})
    df = df[['label', 'text']][: len(df) // 2]
    labels = set(df['label'])

    print(labels)
    df.to_csv(path, index=False)


def load():
    df = pd.read_csv(path)
    print(df.head())



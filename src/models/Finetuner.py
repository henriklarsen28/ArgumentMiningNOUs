
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_dataset_from_csv(csv_path, test_size=0.1):
    """
    Loads a dataset from a CSV file, preprocesses it, and splits it into training and test sets.

    Args:
        csv_path (str): The file path to the CSV file containing the dataset.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' datasets.
    """
    # Load data from CSV
    df = pd.read_csv(csv_path)
    # Assume 'text' and 'label' columns exist; adapt as necessary
    df = df[['text', 'label']].dropna()  # Basic cleaning, dropping rows with NaN values

    # Create a Dataset from pandas DataFrame
    full_dataset = Dataset.from_pandas(df)
    
    # Convert labels to integers
    label2id = {label: i for i, label in enumerate(set(full_dataset['label']))}
    full_dataset = full_dataset.map(lambda example: {'label': label2id[example['label']]})
    
    # Split dataset into train and test
    train_test_split = full_dataset.train_test_split(test_size=test_size)

    return train_test_split


def compute_metrics(eval_pred):
    """Function for computing evaluation metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metric_names = ['accuracy']
    metrics = {}

    for metric in metric_names:
        metrics[metric] = evaluate.load(metric).compute(predictions=predictions, references=labels)[metric]

    return metrics


class FineTuner2:
    def __init__(self, model_name, csv_path, num_epochs=5, max_tokenized_length=128):
        # Load dataset using the new function
        dataset = load_dataset_from_csv(csv_path)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(set(dataset['train']['label'])),
            id2label={i: label for i, label in enumerate(set(dataset['train']['label']))},
            label2id={label: i for i, label in enumerate(set(dataset['train']['label']))})
        self.max_tokenized_length = max_tokenized_length
        self.num_epochs = num_epochs
        self.seed = 42

        # Initialize trainer
        self.trainer = self.init_trainer(dataset)

    def tokenize_function(self, examples):
        return self.tokenizer(
            text=examples["text"],
            padding=True,
            truncation=True,
            max_length=self.max_tokenized_length,
            return_tensors="pt")

    def init_trainer(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"].shuffle(seed=self.seed)
        test_dataset = tokenized_datasets["test"].shuffle(seed=self.seed)

        training_args = TrainingArguments(
            output_dir="../../classifiers" + 'bert-model',
            evaluation_strategy="steps",
            save_strategy='epoch',
            optim='adamw_torch',
            num_train_epochs=self.num_epochs,
            auto_find_batch_size=True)
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()


# Usage:
finetuner = FineTuner2(model_name="NBAiLab/nb-bert-large", csv_path="../../dataset/nou_hearings.csv")
finetuner.train()
results = finetuner.evaluate()
print(results)

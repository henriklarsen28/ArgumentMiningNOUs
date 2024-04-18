import os
from typing import Union, List

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments, pipeline,
)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class FineTuner:
    def __init__(self, model_name: str, csv_path: str,
                 num_epochs: int = 5,
                 max_tokenized_length: int = 512,
                 metric_names: tuple = tuple('accuracy'),
                 wand_logging: bool = True,
                 eval_steps: int = 50):
        # Run time constants
        self.max_tokenized_length = max_tokenized_length
        self.num_epochs = num_epochs
        self.seed = 42
        self.metric_names = metric_names
        self.wandb_logging = wand_logging
        self.eval_steps = eval_steps

        self.device = select_device()
        print(f'Device: {self.device}')

        # Load dataset
        dataset = self.load_dataset_from_csv(csv_path)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(set(dataset['train']['label'])),
            id2label={i: label for i, label in enumerate(set(dataset['train']['label']))},
            label2id={label: i for i, label in enumerate(set(dataset['train']['label']))})

        self.model.to(self.device)

        # Initialize trainer
        self.trainer = self.init_trainer(dataset)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                                   device=self.device.index if self.device.type != 'cpu' else -1)

        # Initialize Weights and Biases
        if self.wandb_logging:
            self.wandb = wandb.init(project="TDT4310-NLP",
                                    config={
                                        'base_model': model_name,
                                        'dataset': dataset['train'].config_name,
                                        'train_dataset_size': len(dataset['train']),
                                        'eval_dataset_size': len(dataset['test']),
                                        'max_tokenized_length': self.max_tokenized_length,
                                    })
        else:
            os.environ["WANDB_DISABLED"] = "true"

    def load_dataset_from_csv(self, csv_path, test_size=0.1):
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

        # Convert labels to integers
        label2id = {label: i for i, label in enumerate(set(full_dataset['label']))}
        full_dataset = full_dataset.map(lambda example: {'label': label2id[example['label']]})

        # Split dataset into train and test
        train_test_split = full_dataset.train_test_split(seed=self.seed, shuffle=True, test_size=test_size)
        return train_test_split

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {}

        for metric in self.metric_names:
            if metric is 'accuracy':
                metrics[metric] = evaluate.load(metric).compute(
                    predictions=predictions, references=labels)[metric]
            else:
                metrics[metric] = evaluate.load(metric).compute(
                    predictions=predictions, references=labels, average='macro')[metric]

        return metrics

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
            output_dir=os.path.join("../../classifiers", 'bert-model'),
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
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
            compute_metrics=self.compute_metrics,
        )

    def classify(self, text):
        # Initialize pipeline
        return self.classifier(text)

    def predict(self, data: Union[str, List[str]]) -> torch.Tensor:
        """
        Generates a prediction for the data and returns probabilities as a tensor.
        """
        encoding = self.tokenizer(data, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = encoding['input_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Ensure batch size is handled correctly
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def train(self):
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()


# Run:
finetuner = FineTuner(model_name="NBAiLab/nb-bert-large",
                      csv_path="../../dataset/nou_hearings.csv",
                      num_epochs=10,
                      metric_names=('accuracy', 'recall', 'precision', 'f1'),
                      wand_logging=True,
                      eval_steps=2)
finetuner.train()
results = finetuner.evaluate()
print(results)

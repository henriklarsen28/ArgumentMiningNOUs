import os

import evaluate
import numpy as np
import torch
import wandb
from datasets import Dataset
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from src.util.helpers import load_dataset_from_csv, configure_for_regression


class FineTuner:
    def __init__(self, model_name: str, csv_path: str, output_folder: str, output_name: str,
                 num_epochs: int = 5,
                 max_tokenized_length: int = 512,
                 wand_logging: bool = True,
                 eval_steps: int = 50,
                 regression=False):
        # Run time constants
        self.max_tokenized_length = max_tokenized_length
        self.num_epochs = num_epochs
        self.seed = 42
        self.wandb_logging = wand_logging
        self.eval_steps = eval_steps
        self.output_folder = output_folder
        self.output_name = output_name
        self.regression = regression

        # Load dataset
        self.dataset, num_labels = load_dataset_from_csv(csv_path)

        # Adjust for regression
        if not regression:
            self.metric_names = ('accuracy', 'recall', 'precision', 'f1')
        else:
            num_labels = 1
            self.metric_names = ('mse',)
            self.dataset['train'] = configure_for_regression(self.dataset['train'])
            self.dataset['test'] = configure_for_regression(self.dataset['test'])

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Initialize trainer
        self.trainer = self.init_trainer(self.dataset)

        # Initialize Weights and Biases
        if self.wandb_logging:
            self.wandb = wandb.init(project="TDT4310-NLP",
                                    config={
                                        'base_model': model_name,
                                        'dataset': self.dataset['train'].config_name,
                                        'train_dataset_size': len(self.dataset['train']),
                                        'eval_dataset_size': len(self.dataset['test']),
                                        'max_tokenized_length': self.max_tokenized_length,
                                    })
        else:
            os.environ["WANDB_DISABLED"] = "true"

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {}

        for metric in self.metric_names:
            if metric == 'accuracy':
                metrics[metric] = evaluate.load(metric).compute(
                    predictions=predictions, references=labels)[metric]
            elif self.regression:
                predictions = logits.squeeze()
                mse = mean_squared_error(labels, predictions)
                metrics['mse'] = mse
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

    def init_trainer(self, dataset: Dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.output_name),
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy='steps',
            save_steps=500,
            optim='adamw_torch',
            num_train_epochs=self.num_epochs,
            auto_find_batch_size=True,
            load_best_model_at_end=False,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_folder, self.output_name+"-Final"))

    def evaluate(self):
        return self.trainer.evaluate()
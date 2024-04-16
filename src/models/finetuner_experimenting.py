from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm


class SequenceClassifier:
    def __init__(self, model_name, csv_path, num_epochs=5, max_tokenized_length=128):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            id2label={i: label for i, label in enumerate(set(dataset['train']['label']))},
            label2id={label: i for i, label in enumerate(set(dataset['train']['label']))}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_epochs = num_epochs
        self.max_tokenized_length = max_tokenized_length

        # Initialize data loading and preparation
        self.label_dict = None  # To store label-index mapping
        self.dataset = None
        self.train_loader = None
        self.load_data(csv_path)

        # Prepare optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def load_data(self, csv_path):
        # Load the dataset
        dataset = load_dataset('csv', data_files=csv_path).train_test_split(test_size=0.3)
        unique_labels = set(dataset['train']['label'])
        id2label = {idx: label for idx, label in enumerate(unique_labels)}
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        # Preprocess the dataset
        self.dataset = dataset.map(self.tokenize_function, batched=True)
        self.dataset = self.dataset.map(self.convert_labels_to_ids, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Prepare the DataLoader
        self.train_loader = DataLoader(self.dataset['train'], batch_size=8, shuffle=True)


    def tokenize_function(self, examples):
        # Tokenize the inputs
        return self.tokenizer(
            examples['text'], padding="max_length", truncation=True, max_length=self.max_tokenized_length
        )

    def convert_labels_to_ids(self, examples):
        # Convert labels using the dynamically created label dictionary
        labels = [self.label_dict[label] for label in examples['label']]
        examples['labels'] = torch.tensor(labels)
        return examples

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            loop = tqdm(self.train_loader, leave=True)
            for batch in loop:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loop.set_description(f'Epoch {epoch + 1}')
                loop.set_postfix(loss=loss.item())


# Example usage
classifier = SequenceClassifier('bert-base-uncased', '../../dataset/nou_hearings.csv', 3, 128)
classifier.train()

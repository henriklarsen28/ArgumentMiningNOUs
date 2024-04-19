import json
import torch


def id2label(label):
    # Read json file

    with open('../../dataset/label2id.json') as f:
        # Convert json file to dictionary
        dicti = json.loads(f.read())
        print(dicti)
        label2id = {label: i for i, label in dicti.items()}
        # Find the value
        label = label2id[label]
        return label


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

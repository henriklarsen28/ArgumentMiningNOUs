from transformers import pipeline
from pandas import DataFrame
from util.helpers import select_device

tokenizer_config = {'padding': True, 'truncation': True, 'max_length': 512}

def predict_dataset(model: str, test_df):
    classifier = pipeline('text-classification', model=model, device=select_device())
    texts = test_df['text'].tolist()
    predictions = classifier(texts, **tokenizer_config)
    predict_column = [prediction['label'][-1] for prediction in predictions]
    test_df['prediction'] = predict_column

    return test_df

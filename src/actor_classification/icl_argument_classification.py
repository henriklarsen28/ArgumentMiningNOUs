from finetuning import train_and_save_classifiers, load_and_evaluate

path = '../../dataset/cleaned_arguments_in_context_learning.csv'


def icl_train():
    train_and_save_classifiers('ICL-Classifier', num_epochs=6, csv_path=path)

# load_and_evaluate('../../classifiers/ICL-Classifier', csv_path=path)


from finetuning import train_and_save_classifiers, load_and_evaluate

path = "../../dataset/LDA_Arguments.csv"


def lda_train():
    train_and_save_classifiers('LDA-Classifier', num_epochs=6, csv_path=path)

# load_and_evaluate('../../classifiers/LDA-Classifier', csv_path=dataset)

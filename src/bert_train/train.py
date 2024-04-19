from finetuning import train_and_save_classifiers


def lda_train():
    path = "../../dataset/LDA_Arguments.csv"
    train_and_save_classifiers('LDA-Classifier', num_epochs=6, csv_path=path)


def icl_train():
    path = '../../dataset/cleaned_arguments_in_context_learning.csv'
    train_and_save_classifiers('ICL-Classifier', num_epochs=6, csv_path=path)


lda_train()
icl_train()

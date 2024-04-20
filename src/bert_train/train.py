from finetuning import train_and_save_classifiers


def lda_train():
    print('Train LDA')
    path = "../../dataset/LDA_Arguments.csv"
    train_and_save_classifiers('LDA-Classifier', num_epochs=6, csv_path=path)


def icl_train():
    print('Train ICL')
    path = '../../dataset/cleaned_arguments_in_context_learning.csv'
    train_and_save_classifiers('ICL-Classifier', num_epochs=6, csv_path=path)


icl_train()

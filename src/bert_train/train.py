from finetuning import train_and_save_classifiers, load_and_evaluate


def raw_train():
    print('Train Raw')
    path = "../../dataset/nou_hearings.csv"
    train_and_save_classifiers('Raw-Classifier', num_epochs=6, csv_path=path)


def raw_eval():
    print('Evaluate Raw')
    path = "../../dataset/nou_hearings.csv"
    load_and_evaluate('../../classifiers/Raw-Classifier-Best', csv_path=path)


def lda_train():
    print('Train LDA')
    path = "../../dataset/LDA_Arguments.csv"
    train_and_save_classifiers('LDA-Classifier', num_epochs=15, csv_path=path)


def lda_eval():
    print('Evaluate LDA')
    path = "../../dataset/LDA_Arguments.csv"
    load_and_evaluate('../../classifiers/LDA-Classifier-Best', csv_path=path)


def icl_train():
    print('Train ICL')
    path = '../../dataset/cleaned_arguments_in_context_learning.csv'
    train_and_save_classifiers('ICL-Classifier', num_epochs=6, csv_path=path)


def icl_eval():
    print('Evaluate ICL')
    path = '../../dataset/cleaned_arguments_in_context_learning.csv'
    load_and_evaluate('../../classifiers/ICL-Classifier-Best', csv_path=path)


def sentiment_train():
    path = '../../dataset/norec.csv'
    train_and_save_classifiers('Sentiment-Classifier', num_epochs=3, csv_path=path, regression=True)


sentiment_train()

from src.models.Finetuner import FineTuner
from raw_paragraph_classification import train_and_save_classifiers, load_and_evaluate

train_and_save_classifiers('ICL-Classifier', num_epochs=6,
                           csv_path="../../dataset/cleaned_arguments_in_context_learning.csv")
# load_and_evaluate('../../classifiers/Raw-Classifier', csv_path='../../dataset/nou_hearings.csv')

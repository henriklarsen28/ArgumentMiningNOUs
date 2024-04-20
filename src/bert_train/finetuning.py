from Finetuner import FineTuner

nbbert = "NbAiLab/nb-bert-large"


def train_and_save_classifiers(output_name, num_epochs, csv_path):
    finetuner = FineTuner(model_name=nbbert,
                          csv_path=csv_path,
                          output_folder="../../classifiers",
                          output_name=output_name,
                          num_epochs=num_epochs,
                          metric_names=('accuracy', 'recall', 'precision', 'f1'),
                          wand_logging=True,
                          eval_steps=30)
    finetuner.train()
    results = finetuner.evaluate()
    print(results)


def load_and_evaluate(model_path, csv_path):
    finetuner = FineTuner(model_name=model_path,
                          csv_path=csv_path,
                          output_folder="../../classifiers",
                          output_name="no-name",
                          num_epochs=1,
                          metric_names=('accuracy', 'recall', 'precision', 'f1'),
                          wand_logging=True,
                          eval_steps=2)
    results = finetuner.evaluate()
    print(results)
from src.models.Finetuner import FineTuner

finetuner = FineTuner(model_name="NBAiLab/nb-bert-large",
                      csv_path="../../dataset/nou_hearings.csv",
                      num_epochs=5,
                      metric_names=('accuracy', 'recall', 'precision', 'f1'),
                      wand_logging=True,
                      eval_steps=50)
finetuner.train()
results = finetuner.evaluate()
print(results)

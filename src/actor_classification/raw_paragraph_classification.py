from finetuning import train_and_save_classifiers, load_and_evaluate


train_and_save_classifiers('Raw-Classifier', num_epochs=6, csv_path="../../dataset/nou_hearings.csv")
load_and_evaluate('../../classifiers/Raw-Classifier', csv_path='../../dataset/nou_hearings.csv')

#train_and_save_classifiers('ICL-Classifier', num_epochs=6, csv_path="../../dataset/nou_hearings.csv")
# train_and_save_classifiers('ICL-Classifier', num_epochs=10,
#                           csv_path="../../dataset/cleaned_arguments_in_context_learning.csv.csv")

# train_and_save_classifiers('ICL-Classifier', num_epochs=5,
#                           csv_path="../../dataset/cleaned_arguments_in_context_learning.csv")
# train_and_save_classifiers('Raw-Classifier', num_epochs=5, csv_path="../../dataset/LDA_Arguments.csv")

# wandb: Run history:
# wandb:           eval/accuracy ▁▃▄▄▅▆▅▆▆▇▇▇▇▇▇▇▇██▇█▇██▇██████▇███
# wandb:                 eval/f1 ▁▃▄▄▅▇▅▆▆▇▇▇▇▇█▇█████▇█████████████
# wandb:               eval/loss █▇▅▅▄▃▄▃▃▂▂▁▂▂▂▃▂▁▁▁▁▂▂▁▂▂▁▂▂▃▃▃▂▂▂
# wandb:          eval/precision ▁▃▅▅▆▇▆▇▇▇██▇████████▇█████████▇███
# wandb:             eval/recall ▁▃▄▄▅▇▅▆▆▇▆▇▇▇▇██████▇█████████████
# wandb:            eval/runtime ▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
# wandb: eval/samples_per_second ▇▇▇▇█▇▇▇▁▇▇▇▇█▇███▇█▇▇█▇██▇▇▇▇▇▇▇██
# wandb:   eval/steps_per_second ▇▇▇▇█▇▇▇▁▇▇▇▇█▇███▇█▇▇█▇██▇▇▇▇▇▇▇██
# wandb:             train/epoch ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████
# wandb:       train/global_step ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████
# wandb:         train/grad_norm ▁▁█
# wandb:     train/learning_rate █▄▁
# wandb:              train/loss █▃▁
# wandb:
# wandb: Run summary:
# wandb:            eval/accuracy 0.82295
# wandb:                  eval/f1 0.68614
# wandb:                eval/loss 0.8555
# wandb:           eval/precision 0.70883
# wandb:              eval/recall 0.67879
# wandb:             eval/runtime 23.3199
# wandb:  eval/samples_per_second 13.079
# wandb:    eval/steps_per_second 1.672
# wandb:               total_flos 1.278165457700352e+16
# wandb:              train/epoch 5.0
# wandb:        train/global_step 1715
# wandb:          train/grad_norm 22.5071
# wandb:      train/learning_rate 1e-05
# wandb:               train/loss 0.3776
# wandb:               train_loss 0.70184
# wandb:            train_runtime 4620.629
# wandb: train_samples_per_second 2.968
# wandb:   train_steps_per_second 0.371

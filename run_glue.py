# -*- coding: utf-8 -*-
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from transformers import TrainingArguments, Trainer

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime
from zipfile import ZipFile

glue_naming_dict = {"cola": 'CoLA.tsv',
                    "mnli": 'MNLI-m.tsv',
                    "mnli-mm": 'MNLI-mm.tsv',
                    "mrpc": 'MRPC.tsv',
                    "qnli": 'QNLI.tsv',
                    "qqp": 'QQP.tsv',
                    "rte": 'RTE.tsv',
                    "sst2": 'SST-2.tsv',
                    "stsb": 'STS-B.tsv',
                    "wnli": 'WNLI.tsv',
                    "ax": 'AX.tsv'}

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 1000)
    }


def run_task(model_dir, task, sample_size, run_mode):
    """Runs a single GLUE-task for a single model. Performs a hyperparameter search over given hyperparameters.
    Contains several nested functions which use other internal variables. """

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    test_key = "test_mismatched" if task == "mnli-mm" else "test_matched" if task == "mnli" else "test"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]

        return metric.compute(predictions=predictions, references=labels)


    def preprocess_function(examples):
        #Helper dictionaries
        task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        }
        sentence1_key, sentence2_key = task_to_keys[task]
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=128)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)

    def compute_objective(metrics):
        return metrics['eval_' + metric_name]

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    save_path = os.path.join(model_dir, 'GLUE', task)

    args = TrainingArguments(output_dir=save_path,
                             overwrite_output_dir=True,
                             evaluation_strategy = "epoch",
                             save_strategy = 'no',
                             metric_for_best_model=metric_name)

    if run_mode == 'full':
        eval_dataset = encoded_dataset[validation_key]

        if task in ['mnli', 'mnli-mm']:
            train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)
        elif task in ['qnli']:
            train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)
        elif task in ['sst2']:
            train_dataset = encoded_dataset["train"].shard(index=1, num_shards=5)
        elif task in ['qqp']:
            train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10)
            eval_dataset = encoded_dataset[validation_key].shard(index=1, num_shards=10)
        else:
            train_dataset = encoded_dataset["train"]

        test_dataset = dataset[test_key].remove_columns('label').map(preprocess_function, batched=True)
    elif run_mode == 'test':
        train_dataset = encoded_dataset["train"].shard(index=1, num_shards=1000)
        eval_dataset = encoded_dataset[validation_key].shard(index=1, num_shards=1000)
        test_dataset = dataset[test_key].remove_columns('label').map(preprocess_function, batched=True).shard(index=1, num_shards=1000)
    else:
        raise ValueError('Invalid argument for variable "run_mode"')

    trainer = Trainer(model_init=model_init,
                      args=args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,  #encoded_dataset[validation_key],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics
    )

    best_run = trainer.hyperparameter_search(n_trials=sample_size,
                                             direction="maximize",
                                             compute_objective=compute_objective,
                                             hp_space=my_hp_space)

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    setattr(trainer, 'train_dataset', encoded_dataset["train"])
    setattr(trainer, 'eval_dataset', encoded_dataset[validation_key])
    trainer.train()

    trainer.save_model(save_path)


    best_model_dev_results = trainer.evaluate()
    best_model_test_results = trainer.predict(test_dataset)


    #Make all predictions for tasks
    if task == 'stsb':
        predictions = np.argmax(F.softmax(torch.tensor(best_model_test_results.predictions), dim=1), axis=1)

    elif task == 'mnli':
        predictions = np.argmax(F.softmax(torch.tensor(best_model_test_results.predictions), dim=1), axis=1)
        label_names = dataset[test_key].features['label'].names
        predictions = [label_names[i] for i in predictions]


        ax_dataset = load_dataset('glue', 'ax')['test']
        ax_encoded_dataset = ax_dataset.remove_columns('label').map(preprocess_function, batched=True)

        ax_test_results = trainer.predict(ax_encoded_dataset)

        ax_predictions = np.argmax(F.softmax(torch.tensor(ax_test_results.predictions), dim=1), axis=1)

        ax_label_names = ax_dataset.features['label'].names
        ax_predictions = [ax_label_names[i] for i in ax_predictions]

    elif task in ['rte', 'qnli']:
        predictions = np.argmax(F.softmax(torch.tensor(best_model_test_results.predictions), dim=1), axis=1)
        label_names = dataset[test_key].features['label'].names
        predictions = [label_names[i] for i in predictions]

    else:
        predictions = np.argmax(F.softmax(torch.tensor(best_model_test_results.predictions), dim=1), axis=1)


    #Store predictions accordingly
    if task == 'mnli':
        df_ax_predictions = pd.DataFrame({'index': np.arange(0, len(ax_predictions)),
                                          'prediction': ax_predictions})
        df_ax_predictions.to_csv(os.path.join(save_path, glue_naming_dict['ax']), sep='\t', index=False, header=True)

    df_predictions = pd.DataFrame({'index': np.arange(0, len(predictions)),
                                   'prediction': predictions})
    df_predictions.to_csv(os.path.join(save_path, glue_naming_dict[task]), sep='\t', index=False, header=True)


    with open(os.path.join(save_path, 'dev_results.pkl'), 'wb') as f:
        pickle.dump(best_model_dev_results, f)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        default="test_experiment/model2/",
                        help='Which pretrained model to finetune')
    parser.add_argument("--sample-size", required=True,
                        type=int,
                        default=2,
                        help="How many trials to perform during hyperparameter search")
    parser.add_argument('--run-mode', required=True,
                        type=str,
                        default='full',
                        help="Whether to run a 1/100 sample or full version of the finetuning.")
    parser.add_argument('--task', required=True,
                        type=str,
                        help='Which tasks to run')
    args = parser.parse_args()

    model_dir = args.model_dir
    sample_size = args.sample_size
    run_mode = args.run_mode

    task  = args.task
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    print("Running task {} at {}".format(task, datetime.now().strftime("%H:%M:%S")))
    run_task(model_dir, task, sample_size, run_mode)
    print("Finished running task {} at {}".format(task, datetime.now().strftime("%H:%M:%S")))


    if task == 'wnli':
        zipObj = ZipFile(os.path.join(model_dir, 'submission.zip'), 'w')

        for task in GLUE_TASKS:
            zipObj.write(os.path.join(model_dir, 'GLUE', task, glue_naming_dict[task]))
            if task == 'mnli':
                zipObj.write(os.path.join(model_dir, 'GLUE', task, glue_naming_dict['ax']))
        zipObj.close()
        print('Saved submission.zip at {}'.format(os.path.join(model_dir, 'submission.zip')))

if __name__ == "__main__":
    main()

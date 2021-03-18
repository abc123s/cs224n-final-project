# heavily inspired by:
# https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews
import os
import subprocess
import json
from datetime import datetime

import torch
import transformers
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from preprocess import preprocess

# transformers.logging.set_verbosity_info()

params = {
    # model information
    "MODEL_NAME": "bert-base-cased",
    
    # training data
    "DATASET": "proprietary",

    # training
    "EPOCHS": 3,
    "BATCH_SIZE": 16,
}


# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)
os.mkdir(experiment_dir + "/logs")
os.mkdir(experiment_dir + "/results")

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# preprocess data
train_dataset, dev_dataset, tokenizer, num_labels = preprocess(
    params["MODEL_NAME"],
    params["DATASET"],
)

# make model
model = BertForSequenceClassification.from_pretrained(params["MODEL_NAME"], num_labels=num_labels)

# make custom metric to track tag-level and sentence-level accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    correct = 0
    total = 0
    for label, pred in zip(labels, preds):
        total += 1

        if label == pred:
            correct += 1

    return {
        "accuracy": correct / total,
    }

# train model
training_args = TrainingArguments(
    output_dir= experiment_dir + '/results',
    num_train_epochs=params["EPOCHS"],
    per_device_train_batch_size=params["BATCH_SIZE"],
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=experiment_dir + '/logs',           
    logging_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


# evaluate model on train and dev set
train_results = trainer.evaluate(train_dataset)
dev_results = trainer.evaluate()

# save down results
with open(experiment_dir + "/results.json", "w") as f:
    json.dump({
        "train_results": train_results,
        "dev_results": dev_results,
    }, f, indent=4)

# save down model for later usage
tokenizer.save_pretrained(experiment_dir)
model.save_pretrained(experiment_dir)


# script to reevaluate bert model performance in a way that is
# comparable to the lstm model

import transformers
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer

from preprocess import preprocess

# transformers.logging.set_verbosity_info()

experiment_dir = "experiments/20210318_0100_c997fdd"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

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

        if label == pred and label != 0:
            correct += 1

    return {
        "accuracy": correct / total,
    }

# build model
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

# evaluate model on train and dev set
train_results = trainer.evaluate(train_dataset)
dev_results = trainer.evaluate()

# save down results
with open(experiment_dir + "/results_no_unmapped.json", "w") as f:
    json.dump({
        "train_results": train_results,
        "dev_results": dev_results,
    }, f, indent=4)

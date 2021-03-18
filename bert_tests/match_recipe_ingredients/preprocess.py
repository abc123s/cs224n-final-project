# heavily inspired by:
# https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews

import json

import numpy as np
import torch
from transformers import BertTokenizerFast


class IngredientMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# helper function to split example pairs into separate
# text and tag lists
def split_examples(examples):
    texts = [example[0] for example in examples]
    labels = [int(example[1]) for example in examples]

    return texts, labels

def preprocess(model_name, dataset):
    with open(f"./data/{dataset}/training_examples.json", "r") as f:
        train_examples = json.load(f)

    train_texts, train_labels = split_examples(train_examples)

    with open(f"./data/{dataset}/dev_examples.json", "r") as f:
        dev_examples = json.load(f)

    dev_texts, dev_labels = split_examples(dev_examples)

    # encode text
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, padding=True, truncation=True)
    dev_encodings = tokenizer(dev_texts, padding=True, truncation=True)

    num_labels = max([*train_labels, *dev_labels]) + 1

    # prepare datasets for consumption by model
    train_dataset = IngredientMatchingDataset(train_encodings, train_labels)
    dev_dataset = IngredientMatchingDataset(dev_encodings, dev_labels)

    return train_dataset, dev_dataset, tokenizer, num_labels

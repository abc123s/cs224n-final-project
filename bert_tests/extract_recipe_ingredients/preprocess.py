# heavily inspired by:
# https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities

import json

import numpy as np
import torch
from transformers import BertTokenizerFast


# helper function to align word tags with subword tokenization
# used by BERT
def encode_tags(tag_ids, encodings):
    encoded_tags = []
    for doc_tags, doc_offset in zip(tag_ids, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_tags = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set tags whose first offset position is 0 and the second is not 0
        # (i.e. ignore all subword tokens that are not the first part of the word)
        doc_enc_tags[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_tags
        encoded_tags.append(doc_enc_tags.tolist())

    return encoded_tags


class IngredientTaggingDataset(torch.utils.data.Dataset):
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
    tags = [example[1] for example in examples]

    return texts, tags

def preprocess(model_name, dataset):
    with open(f"./data/{dataset}/training_examples.json", "r") as f:
        train_examples = json.load(f)

    train_texts, train_tags = split_examples(train_examples)

    train_texts = train_texts[0:32]
    train_tags = train_tags[0:32]

    with open(f"./data/{dataset}/dev_examples.json", "r") as f:
        dev_examples = json.load(f)

    dev_texts, dev_tags = split_examples(dev_examples)

    # encode text
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    dev_encodings = tokenizer(dev_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

    # encode tags
    unique_tags = sorted(set(tag for doc in dev_tags for tag in doc))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    train_tag_ids = [[tag2id[tag] for tag in doc] for doc in train_tags]
    dev_tag_ids = [[tag2id[tag] for tag in doc] for doc in dev_tags]
    
    train_labels = encode_tags(train_tag_ids, train_encodings)
    dev_labels = encode_tags(dev_tag_ids, dev_encodings)

    # prepare datasets for consumption by model
    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    dev_encodings.pop("offset_mapping")
    train_dataset = IngredientTaggingDataset(train_encodings, train_labels)
    dev_dataset = IngredientTaggingDataset(dev_encodings, dev_labels)

    return train_dataset, dev_dataset, tokenizer, tag2id, id2tag

'''
Preprocess manually tagged data
'''

import json

from preprocess_simple import build_encodings, build_dataset

from tokenizer import IngredientPhraseTokenizer

ingredientPhraseTokenizer = IngredientPhraseTokenizer()

def load_examples(data_path, dataset_name):
    examples_path = data_path + f"/{dataset_name}_examples.json"

    # load examples
    with open(examples_path, "r") as f:
        examples = json.load(f)

    return examples


def preprocess(data_path, examples_for_vocab=None, pretrained_embeddings=None, embedding_size=None):
    train_examples = load_examples(data_path, "manually_tagged_train")
    dev_examples = load_examples(data_path, "manually_tagged_dev")

    tokenized_train_examples = [
        ingredientPhraseTokenizer.tokenize(train_example[0])
        for train_example in train_examples
    ]

    tokenized_dev_examples = [
        ingredientPhraseTokenizer.tokenize(dev_example[0])
        for dev_example in dev_examples
    ]

    if examples_for_vocab:
        word_encoder, tag_encoder = build_encodings(examples_for_vocab, pretrained_embeddings, embedding_size)
    else:
        word_encoder, tag_encoder = build_encodings(train_examples, pretrained_embeddings, embedding_size)

    train_dataset = build_dataset(train_examples, word_encoder, tag_encoder)
    dev_dataset = build_dataset(dev_examples, word_encoder, tag_encoder)

    return train_dataset, dev_dataset, None, tokenized_train_examples, tokenized_dev_examples, word_encoder, tag_encoder

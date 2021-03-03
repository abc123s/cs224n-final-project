import os
import json

import tensorflow as tf
import numpy as np

from preprocessing.preprocess import preprocess_test

from model.model import build_model

# ingredient dictionary to match to
with open(os.path.join(os.path.dirname(__file__), "data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

# train examples to evaluate
with open(os.path.join(os.path.dirname(__file__), "data/trainMatchedTrainingExamples.json")) as train_examples_data:
    train_examples = json.load(train_examples_data)

# test examples to evaluate
with open(os.path.join(os.path.dirname(__file__), "data/devMatchedTrainingExamples.json")) as test_examples_data:
    test_examples = json.load(test_examples_data)

# map each test example to the ingredient dictionary entry with the top 5 closest
# embeddings to the training example's embedding
def select_ingredient_dictionary_match(embedded_example, embedded_ingredient_dictionary):
    ranked_ingredient_dictionary_entries = sorted(
        embedded_ingredient_dictionary.items(),
        key = lambda dictionary_entry: np.mean((dictionary_entry[1] - embedded_example) ** 2)
    )

    return [ingredient_id for ingredient_id, _ in ranked_ingredient_dictionary_entries[0: 5]]

def accuracy(examples, embedding, embedded_ingredient_dictionary, use_tags):
    # compute embeddings of examples
    raw_examples = (
        [[example["original"], example["tags"]] for example in examples]
        if use_tags
        else [example["original"] for example in examples]
    )
    example_embeddings = [
        example_embedding.numpy() 
        for example_embedding in embedding(
            preprocess_test(raw_examples),
            training=False
        )
    ]

    # match examples to ingredient dictionary based on distance of embeddings
    example_preds = [
        select_ingredient_dictionary_match(embedded_example, embedded_ingredient_dictionary) 
        for embedded_example in example_embeddings
    ]

    # evalute accuracy of matches
    example_labels = [
        example["ingredients"][0]["ingredient"] and example["ingredients"][0]["ingredient"]["id"] 
        for example in examples
    ]
    total_examples = 0
    correct_examples = 0
    top_3_correct_examples = 0
    top_5_correct_examples = 0
    for pred, label in zip(example_preds, example_labels):
        total_examples += 1
        
        if pred[0] == label:
            correct_examples += 1
        
        if (label in pred[0:3]):
            top_3_correct_examples += 1
        
        if (label in pred):
            top_5_correct_examples += 1            

    return {
        "accuracy": correct_examples / total_examples,
        "top-3 accuracy": top_3_correct_examples / total_examples,
        "top-5 accuracy": top_5_correct_examples / total_examples,
    }

def evaluate(model):
    # determine model whether embedding takes tags as input as well or just tokens
    use_tags = True
    try:
        model.get_layer('anchor_tags')
    except:
        use_tags = False

    embedding = model.get_layer('embedding')

    with tf.device('/CPU:0'):
        # compute embeddings of all ingredient dictionary entries for comparison
        # flatten ingredient dictionary
        flat_ingredient_dictionary = list(ingredient_dictionary.items())
        ingredient_dictionary_ids = [ingredient_id for ingredient_id, _ in flat_ingredient_dictionary]
        ingredient_dictionary_entries = (
            [[entry, None] for _, entry in flat_ingredient_dictionary]
            if use_tags
            else [entry for _, entry in flat_ingredient_dictionary]
        )

        # compute embeddings of ingredient dictionary entries
        ingredient_dictionary_entry_embeddings = embedding(
            preprocess_test(ingredient_dictionary_entries),
            training = False
        )

        # reassemble into dictionary
        embedded_ingredient_dictionary = {}
        for ingredient_id, entry_embedding in zip(ingredient_dictionary_ids, ingredient_dictionary_entry_embeddings):
            embedded_ingredient_dictionary[ingredient_id] = entry_embedding.numpy()

        return {
            "train": accuracy(train_examples, embedding, embedded_ingredient_dictionary, use_tags),
            "test": accuracy(test_examples, embedding, embedded_ingredient_dictionary, use_tags),
        }
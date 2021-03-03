import os
import json
import csv

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer, TagTokenizer
from preprocessing.preprocess import preprocess_test

from model.model import build_model

ingredientPhraseTokenizer = IngredientPhraseTokenizer()
tagTokenizer = TagTokenizer()
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

# create word_encoder (to get vocab size)
with open(os.path.join(os.path.dirname(__file__), "preprocessing/vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

# create tag_encoder (to get tag vocab size)
with open(os.path.join(os.path.dirname(__file__), "preprocessing/tag_list.json")) as tag_list_data:
    tag_list = json.load(tag_list_data)

tag_encoder = TokenTextEncoder(tag_list, oov_buckets=0, tokenizer=tagTokenizer)


# ingredient dictionary to match to
with open(os.path.join(os.path.dirname(__file__), "data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

# train examples to evaluate
with open(os.path.join(os.path.dirname(__file__), "data/trainMatchedTrainingExamples.json")) as train_examples_data:
    train_examples = json.load(train_examples_data)

# test examples to evaluate
with open(os.path.join(os.path.dirname(__file__), "data/devMatchedTrainingExamples.json")) as test_examples_data:
    test_examples = json.load(test_examples_data)

# map each test example to the ingredient dictionary entry with the closest
# embeddings to the training example's embedding
def select_ingredient_dictionary_match(embedded_example, embedded_ingredient_dictionary):
    ranked_ingredient_dictionary_entries = sorted(
        embedded_ingredient_dictionary.items(),
        key = lambda dictionary_entry: np.mean((dictionary_entry[1] - embedded_example) ** 2)
    )

    return (
        ranked_ingredient_dictionary_entries[0][0],
        np.mean((ranked_ingredient_dictionary_entries[0][1] - embedded_example) ** 2)
    )

    
def grade(examples, embedding, embedded_ingredient_dictionary, use_tags, training_count):
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
    example_preds = []
    example_pred_dists = []
    for embedded_example in example_embeddings:
        match, dist = select_ingredient_dictionary_match(embedded_example, embedded_ingredient_dictionary)
        example_preds.append(match)
        example_pred_dists.append(dist)

    # evalute accuracy of matches
    example_labels = [
        example["ingredients"][0]["ingredient"] and example["ingredients"][0]["ingredient"]["id"] 
        for example in examples
    ]

    # get id of training example (to fix mis-taggings)
    example_ids = [example["id"] for example in examples]

    # get whether each of the training example matches is "not exact"
    example_not_exacts = [
        example["ingredients"][0].get("notExact", False) if example["ingredients"][0]["ingredient"] else False
        for example in examples
    ]

    correct_examples = []
    incorrect_examples = []
    for id, raw, pred, label, dist, not_exact in zip(example_ids, raw_examples, example_preds, example_labels, example_pred_dists, example_not_exacts):        
        if pred == label:
            correct_examples.append({
                "id": id,
                "raw": raw[0] if use_tags else raw,
                "pred": ingredient_dictionary[pred],
                "label": label and ingredient_dictionary[label],
                "training_count": training_count.get(label, 0),
                "dist": dist,
                "not_exact": not_exact
            })
        else:
            incorrect_examples.append({
                "id": id,
                "raw": raw[0] if use_tags else raw,
                "pred": ingredient_dictionary[pred],
                "label": label and ingredient_dictionary[label],
                "training_count": training_count.get(label, 0),
                "dist": dist,
                "not_exact": not_exact
            })
    return {
        "correct": correct_examples,
        "incorrect": incorrect_examples,
    }

def write_results(results, file_name):
    with open(experiment_dir + '/' + file_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(["id", "raw", "pred", "label", "dist", "training_count", "not_exact"])

        for result in results:
            writer.writerow([result["id"], result["raw"], result["pred"], result["label"], result["dist"], result["training_count"], result["not_exact"]])

def error_analysis(model):
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

        # count occurences of labels in training data
        training_example_labels = [
            example["ingredients"][0]["ingredient"] and example["ingredients"][0]["ingredient"]["id"] 
            for example in train_examples
        ]

        # count number of occurrences of this label in the training data
        training_count = {
            label: training_example_labels.count(label)
            for label in training_example_labels
        }

        train_results = grade(train_examples, embedding, embedded_ingredient_dictionary, use_tags, training_count)
        test_results = grade(test_examples, embedding, embedded_ingredient_dictionary, use_tags, training_count)

        write_results(train_results["correct"], "train_correct")
        write_results(train_results["incorrect"], "train_incorrect")
        write_results(test_results["correct"], "test_correct")
        write_results(test_results["incorrect"], "test_incorrect")

        '''
        train_results_mistake_sample = np.random.choice(train_results["incorrect"], size = 50, replace = False)
        test_results_mistake_sample = np.random.choice(test_results["incorrect"], size = 50, replace = False)

        print('sample train errors:')
        for mistake in train_results_mistake_sample:
            print(mistake["raw"], mistake["pred"], mistake["label"], mistake["dist"], mistake["training_count"])

        print('sample test errors:')
        for mistake in test_results_mistake_sample:
            print(mistake["raw"], mistake["pred"], mistake["label"], mistake["dist"], mistake["training_count"])
        '''

experiment_dir = "experiments/20201203_1651_63cfe44"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# build and compile model based on experiment params:
model, _ = build_model(
    vocab_size = word_encoder.vocab_size,
    word_embedding_size = params["WORD_EMBEDDING_SIZE"],
    sentence_embedding_size = params["SENTENCE_EMBEDDING_SIZE"],
    embedding_architecture = params["EMBEDDING_ARCHITECTURE"],
    triplet_margin = params["TRIPLET_MARGIN"],
    kernel_regularizer = params["KERNEL_REGULARIZER"],
    kernel_regularization_factor = params["REGULARIZATION_FACTOR"],
    recurrent_regularizer = params["RECURRENT_REGULARIZER"],
    recurrent_regularization_factor = params["RECURRENT_REGULARIZATION_FACTOR"],
    dropout_rate = params["DROPOUT_RATE"],
    recurrent_dropout_rate = params["RECURRENT_DROPOUT_RATE"],
    use_tags = params["USE_TAGS"],
    embed_tags = params["EMBED_TAGS"],
    tag_size = tag_encoder.vocab_size,
    tag_embedding_size = params["TAG_EMBEDDING_SIZE"]
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# perform error analysis on model
error_analysis(model)

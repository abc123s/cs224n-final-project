import os
import json
import random

import numpy as np

from preprocessing.preprocess import preprocess_test, preprocess_train_batch

# ingredient dictionary to match to
with open(os.path.join(os.path.dirname(__file__), "../data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

# raw training examples (not yet made into triplets)
with open(os.path.join(os.path.dirname(__file__), "../data/trainMatchedTrainingExamples.json")) as training_examples_data:
    training_examples = json.load(training_examples_data)    

# group raw training examples by ingredient_id (label) and remove unnecessary info
clean_training_examples = []
for training_example in training_examples:
    ingredient_id = (
        training_example["ingredients"][0]["ingredient"] and 
        training_example["ingredients"][0]["ingredient"]["id"]
    )

    clean_training_examples.append({
        "text": training_example["original"],
        "ingredient_id": ingredient_id,
    })

# generate 50 hard and 50 semi-hard training examples for each ingredient_id
# as in FaceNet, select from all positive examples, but choose
# hard or semi-hard negative examples
def generate_batch(model, batch_size, triplets_per_example, include_no_match, margin, pretrained_embeddings = None, embedding_size = None):
    embedding = model.get_layer('embedding')

    # final cleaning of training examples
    final_training_examples = clean_training_examples
    if not include_no_match:
        final_training_examples = [
            example 
            for example in final_training_examples
            if example["ingredient_id"] == None
        ]

    # compute embeddings of all ingredient dictionary entries for comparison
    # flatten ingredient dictionary
    flat_ingredient_dictionary = list(ingredient_dictionary.items())
    ingredient_dictionary_ids = [ingredient_id for ingredient_id, _ in flat_ingredient_dictionary]
    ingredient_dictionary_entries = [entry for _, entry in flat_ingredient_dictionary]

    # compute embeddings of ingredient dictionary entries
    ingredient_dictionary_entry_embeddings = embedding(preprocess_test(
        ingredient_dictionary_entries,
        pretrained_embeddings,
        embedding_size
    ))

    # reassemble into dictionary
    embedded_ingredient_dictionary = {}
    for ingredient_id, entry_embedding in zip(ingredient_dictionary_ids, ingredient_dictionary_entry_embeddings):
        embedded_ingredient_dictionary[ingredient_id] = entry_embedding.numpy()

    # select hard triplets
    batch_triplets = []
    while len(batch_triplets) == 0:
        # select batch of training examples
        training_example_batch = np.random.choice(
            clean_training_examples,
            size = batch_size,
            replace = False,
        )

        # compute embeddings of selected training examples
        batch_text = [training_example["text"] for training_example in training_example_batch]
        batch_ingredient_ids = [training_example["ingredient_id"] for training_example in training_example_batch]
        batch_embeddings = embedding(preprocess_test(
            batch_text,
            pretrained_embeddings,
            embedding_size
        ))

        for example_text, example_ingredient_id, example_embedding in zip(batch_text, batch_ingredient_ids, batch_embeddings):
            anchor = example_text

            # compute distance between example and ingredient dictionary entries
            ingredient_dictionary_dist = np.mean((ingredient_dictionary_entry_embeddings - example_embedding) ** 2, axis = 1)

            # identify positive example (matching ingredient dictionary entry)
            # and its embedding's distance from the anchor
            if example_ingredient_id != None:
                positive_index = ingredient_dictionary_ids.index(example_ingredient_id)
                positive = ingredient_dictionary_entries[positive_index]
                positive_dist = ingredient_dictionary_dist[positive_index]
            # handle special case of no match
            else:
                positive_index = None
                positive = anchor
                positive_dist = 0

            # categorize ingredient dictionary entries into hard and semi-hard
            # hard: closer to example_embedding than correct label
            # semi-hard: further from example_embedding tha correct label, but less than margin
            hard_negatives = []
            semi_hard_negatives = []
            for index, (ingredient_dictionary_entry, dist) in enumerate(zip(ingredient_dictionary_entries, ingredient_dictionary_dist)):
                # skip over the positive ingredient dictionary entry
                if index != positive_index:
                    if dist < positive_dist:
                        hard_negatives.append(ingredient_dictionary_entry)
                    elif dist < positive_dist + margin:
                        semi_hard_negatives.append(ingredient_dictionary_entry)
                
            # construct triplets for this example, trying to evenly split between hard and semi-hard if possible
            hard_negatives_to_select = min(
                (
                    int(triplets_per_example / 2) 
                    if len(semi_hard_negatives) > triplets_per_example / 2
                    else triplets_per_example - len(semi_hard_negatives)
                ),
                len(hard_negatives)
            )
            semi_hard_negatives_to_select = min(
                (
                    int(triplets_per_example / 2)
                    if len(hard_negatives) > triplets_per_example / 2
                    else triplets_per_example - len(hard_negatives)
                ),
                len(semi_hard_negatives)
            )
            selected_negatives = [
                *np.random.choice(hard_negatives, size = hard_negatives_to_select, replace = False),
                *np.random.choice(semi_hard_negatives, size = semi_hard_negatives_to_select, replace = False)
            ]
            batch_triplets.extend([[anchor, positive, negative] for negative in selected_negatives])
    
    return preprocess_train_batch(
        batch_triplets,
        pretrained_embeddings,
        embedding_size
    )

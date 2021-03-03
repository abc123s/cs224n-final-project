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
grouped_training_examples = {
    ingredient_id: [] for ingredient_id, _ in ingredient_dictionary.items()
}
grouped_training_examples[None] = []

for training_example in training_examples:
    ingredient_id = (
        training_example["ingredients"][0]["ingredient"] and 
        training_example["ingredients"][0]["ingredient"]["id"]
    )

    grouped_training_examples[ingredient_id].append(training_example["original"])

# generate 50 hard and 50 semi-hard training examples for each ingredient_id
# as in FaceNet, select from all positive examples, but choose
# hard or semi-hard negative examples
def generate_batch(model, ingredient_ids, margin):
    embedding = model.get_layer('embedding')

    # compute embeddings for all examples in batch as well as embeddings
    # for associated ingredient dictionary entries
    batch_examples = []
    batch_ingredient_dictionary_embeddings = {}
    for ingredient_id in ingredient_ids:
        # grab ingredient dictionary entry
        ingredient_name = ingredient_dictionary[ingredient_id]

        # grab examples for ingredient_id
        ingredient_examples = grouped_training_examples[ingredient_id]

        # compute embeddings for both of the above
        embedded_ingredient_name, *embedded_ingredient_examples = embedding(preprocess_test([
            ingredient_name,
            *ingredient_examples,
        ]))

        # store embeddings of ingredient dictionary entries
        batch_ingredient_dictionary_embeddings[ingredient_id] = embedded_ingredient_name.numpy()

        # store embeddings of examples
        batch_examples.extend(zip(
            [ingredient_id] * len(ingredient_examples),
            ingredient_examples,
            [example.numpy() for example in embedded_ingredient_examples],
        ))

    hard_triplet_examples = []
    for ingredient_id in ingredient_ids:
        anchor = ingredient_dictionary[ingredient_id]
        anchor_embedding = batch_ingredient_dictionary_embeddings[ingredient_id]

        # identify possible positive and negative examples
        positive_examples = [example for example in batch_examples if example[0] == ingredient_id]
        negative_examples = [example for example in batch_examples if example[0] != ingredient_id]

        if len(positive_examples) and len(negative_examples):
            # compute distance between positive examples and anchor
            positive_example_embeddings = np.array([example[2] for example in positive_examples])
            positive_example_anchor_dist = np.mean((positive_example_embeddings - anchor_embedding) ** 2, axis = 1)
            positive_examples = list(zip(positive_examples, positive_example_anchor_dist))

            # compute distance between negative examples and anchor
            negative_example_embeddings = np.array([example[2] for example in negative_examples])
            negative_example_anchor_dist = np.mean((negative_example_embeddings - anchor_embedding) ** 2, axis = 1)
            negative_examples = list(zip(negative_examples, negative_example_anchor_dist))

            for _ in range(50):
                selected_positive_example, positive_anchor_dist = random.choice(positive_examples)

                hard_negative_examples = [
                    negative_example for negative_example, negative_anchor_dist in negative_examples
                    if negative_anchor_dist < positive_anchor_dist
                ]

                semi_hard_negative_examples = [
                    negative_example for negative_example, negative_anchor_dist in negative_examples
                    if negative_anchor_dist > positive_anchor_dist and negative_anchor_dist < positive_anchor_dist + margin
                ]

                # select 1 hard and 1 semi-hard if possible; otherwise select 2 semi-hard
                # or 2 hard if forced to do so
                # if there are no hard or semi-hard negative examples, skip, since easy
                # examples have zero loss
                if len(hard_negative_examples) or len(semi_hard_negative_examples):
                    selected_hard_negative_example = (
                        random.choice(hard_negative_examples) 
                            if len(hard_negative_examples)
                            else random.choice(semi_hard_negative_examples)
                    )
                    hard_triplet_examples.append([
                        anchor,
                        selected_positive_example[1],
                        selected_hard_negative_example[1],
                    ])

                    selected_semi_hard_negative_example = (
                        random.choice(semi_hard_negative_examples) 
                            if len(semi_hard_negative_examples)
                            else random.choice(hard_negative_examples)
                    )
                    hard_triplet_examples.append([
                        anchor,
                        selected_positive_example[1],
                        selected_semi_hard_negative_example[1],
                    ])

    return preprocess_train_batch(hard_triplet_examples)



            

import os
import json

import numpy as np

import tensorflow_datasets as tfds

from preprocessing.preprocess import preprocess_test
from preprocessing.tokenizer import IngredientPhraseTokenizer

from model.model import build_model

ingredientPhraseTokenizer = IngredientPhraseTokenizer()
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

with open(os.path.join(os.path.dirname(__file__), "preprocessing/vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

# ingredient dictionary to match to
with open(os.path.join(os.path.dirname(__file__), "data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

experiment_dir = "experiments/20201118_1405_3cc2903"

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
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# map embedding to the ingredient dictionary entry with the top 5 closest
# embeddings to the specified embedding
def select_ingredient_dictionary_match(embedding, embedded_ingredient_dictionary):
    print(embedding)
    ranked_ingredient_dictionary_entries = sorted(
        embedded_ingredient_dictionary.items(),
        key = lambda dictionary_entry: np.mean((dictionary_entry[1] - embedding) ** 2)
    )

    return [ingredient_id for ingredient_id, _ in ranked_ingredient_dictionary_entries[0: 5]]

embedding = model.get_layer('embedding')

# compute embeddings of all ingredient dictionary entries for comparison
# flatten ingredient dictionary
flat_ingredient_dictionary = list(ingredient_dictionary.items())
ingredient_dictionary_ids = [ingredient_id for ingredient_id, _ in flat_ingredient_dictionary]
ingredient_dictionary_entries = [entry for _, entry in flat_ingredient_dictionary]

# compute embeddings of ingredient dictionary entries
ingredient_dictionary_entry_embeddings = embedding(
    preprocess_test(ingredient_dictionary_entries),
    training = False
)

# reassemble into dictionary
embedded_ingredient_dictionary = {}
for ingredient_id, entry_embedding in zip(ingredient_dictionary_ids, ingredient_dictionary_entry_embeddings):
    embedded_ingredient_dictionary[ingredient_id] = entry_embedding.numpy()

test_ingredient_dictionary_entries = [1, 159, 50]

print('Top 5 ingredient dictionary entries near russet potato:')
for ingredient_id in select_ingredient_dictionary_match(embedded_ingredient_dictionary["1"], embedded_ingredient_dictionary):
    print(ingredient_dictionary[ingredient_id])
    print(np.mean((embedded_ingredient_dictionary["1"] - embedded_ingredient_dictionary[ingredient_id]) ** 2))

print('Top 5 ingredient dictionary entries near milk:')
for ingredient_id in select_ingredient_dictionary_match(embedded_ingredient_dictionary["159"], embedded_ingredient_dictionary):
    print(ingredient_dictionary[ingredient_id])
    print(np.mean((embedded_ingredient_dictionary["159"] - embedded_ingredient_dictionary[ingredient_id]) ** 2))

print('Top 5 ingredient dictionary entries near skinless chicken breast:')
for ingredient_id in select_ingredient_dictionary_match(embedded_ingredient_dictionary["50"], embedded_ingredient_dictionary):
    print(ingredient_dictionary[ingredient_id])
    print(np.mean((embedded_ingredient_dictionary["50"] - embedded_ingredient_dictionary[ingredient_id]) ** 2))
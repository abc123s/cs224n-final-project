import json
import os
import random
from datetime import datetime
import subprocess
import math

import numpy as np

import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer
from model.model import build_model
from evaluate import evaluate
from triplet_mining import triplet_mining_batch_by_example, triplet_mining_batch_by_ingredient

# figure out vocab size so we can reconstruct the model
ingredientPhraseTokenizer = IngredientPhraseTokenizer()
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

with open(os.path.join(os.path.dirname(__file__), "preprocessing/vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

# ingredient dictionary
with open(os.path.join(os.path.dirname(__file__), "data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

# make new experiment directory
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# hyperparameters for re-training
ORIGINAL_EXPERIMENT_DIR = "experiments/20201123_1005_1c402b3"
RETRAINING_METHOD = "online_hard_triplet_mining_batch_by_example"
RETRAINING_EPOCHS = 2
# batch by ingredient params
# INGREDIENT_IDS_PER_BATCH = 60
# RETRAINING_STEPS_PER_EPOCH = math.ceil(len(ingredient_dictionary.keys()) / INGREDIENT_IDS_PER_BATCH)
# batch by example params
BATCH_SIZE = 100
TRIPLETS_PER_EXAMPLE = 20
INGREDIENT_IDS_PER_BATCH = None # not relevant for batch by example
RETRAINING_STEPS_PER_EPOCH = 8000 / BATCH_SIZE # around 1 pass through dataset per EPOCH
INCLUDE_NO_MATCH = False

# load original experiment params
with open(ORIGINAL_EXPERIMENT_DIR + "/params.json", "r") as f:
    old_params = json.load(f)

# make list of params (combine old and new parameters)
params = {
    **old_params,
    "ORIGINAL_EXPERIMENT_DIR": ORIGINAL_EXPERIMENT_DIR,
    "RETRAINING_METHOD": RETRAINING_METHOD,
    "RETRAINING_EPOCHS": RETRAINING_EPOCHS,
    "INGREDIENT_IDS_PER_BATCH": INGREDIENT_IDS_PER_BATCH,
    "BATCH_SIZE": BATCH_SIZE,
    "TRIPLETS_PER_EXAMPLE": TRIPLETS_PER_EXAMPLE,
    "RETRAINING_STEPS_PER_EPOCH": RETRAINING_STEPS_PER_EPOCH,
    "INCLUDE_NO_MATCH": INCLUDE_NO_MATCH,
}

# build and compile model based on original experiment params:
model, loss = build_model(
    word_encoder = word_encoder,
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
)

model.compile(loss=loss, optimizer=params["OPTIMIZER"])

# load final weights from original experiment into model:
model.load_weights(ORIGINAL_EXPERIMENT_DIR + "/model_weights")

# generator re-training examples
def batch_generator():
    if RETRAINING_METHOD == "online_hard_triplet_mining_batch_by_ingredient":
        while True:
            selected_ingredient_ids = random.sample(ingredient_dictionary.keys(), INGREDIENT_IDS_PER_BATCH)
            batch = triplet_mining_batch_by_ingredient.generate_batch(
                model, 
                selected_ingredient_ids, 
                params["TRIPLET_MARGIN"]
            )
            if batch[0].shape[0]:
                yield batch, np.zeros(batch[0].shape[0])
    elif RETRAINING_METHOD == "online_hard_triplet_mining_batch_by_example":
        while True:
            batch = triplet_mining_batch_by_example.generate_batch(
                model,
                batch_size = params["BATCH_SIZE"],
                triplets_per_example = params["TRIPLETS_PER_EXAMPLE"],
                include_no_match = params["INCLUDE_NO_MATCH"],
                margin = params["TRIPLET_MARGIN"]
            )
            if batch[0].shape[0]:
                yield batch, np.zeros(batch[0].shape[0])

# retrain model
history = model.fit(
    batch_generator(),
    steps_per_epoch=RETRAINING_STEPS_PER_EPOCH,
    epochs=RETRAINING_EPOCHS
)

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(model)

# evaluate model and save metrics:
evaluation = evaluate(model)

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent=4)

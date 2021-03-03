import json
import os
import random
from datetime import datetime
import subprocess
import math

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer
from model.model import build_model
from evaluate import evaluate
from triplet_mining import triplet_mining_batch_by_example, triplet_mining_batch_by_ingredient

# figure out vocab size so we can construct the model
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

# hyperparameters
params = {}
# model structure
params["WORD_EMBEDDING_SIZE"] = 64
params["PRETRAINED_EMBEDDINGS"] = None
params["SENTENCE_EMBEDDING_SIZE"] = 64
params["EMBEDDING_ARCHITECTURE"] = 'simple'

# regularization
params["KERNEL_REGULARIZER"] = None
params["REGULARIZATION_FACTOR"] = 0
params["RECURRENT_REGULARIZER"] = None
params["RECURRENT_REGULARIZATION_FACTOR"] = 0
params["DROPOUT_RATE"] = 0
params["RECURRENT_DROPOUT_RATE"] = 0

# loss function
params["TRIPLET_MARGIN"] = 0.2

# training
params["OPTIMIZER"] = 'adam'
params["EPOCHS"] = 200
params["TRAINING_METHOD"] = "triplet_mining_batch_by_example"
if params["TRAINING_METHOD"] == "triplet_mining_batch_by_example":
    params["BATCH_SIZE"] = 100
    params["TRIPLETS_PER_EXAMPLE"] = 20
    params["STEPS_PER_EPOCH"] = math.ceil(8000 / params["BATCH_SIZE"]) # around 1 pass through dataset per EPOCH
    params["INCLUDE_NO_MATCH"] = False
    def generate_batch(model):
        return triplet_mining_batch_by_example.generate_batch(
            model,
            batch_size = params["BATCH_SIZE"],
            triplets_per_example = params["TRIPLETS_PER_EXAMPLE"],
            include_no_match = params["INCLUDE_NO_MATCH"],
            margin = params["TRIPLET_MARGIN"]
        )
        return batch
elif params["TRAINING_METHOD"] == "triplet_mining_batch_by_ingredient":
    params["INGREDIENT_IDS_PER_BATCH"] = 60
    params["STEPS_PER_EPOCH"] = math.ceil(len(ingredient_dictionary.keys()) / params["INGREDIENT_IDS_PER_BATCH"])
    def generate_batch(model):
        return triplet_mining_batch_by_ingredient.generate_batch(
            model, 
            ingredient_ids = random.sample(ingredient_dictionary.keys(), params["INGREDIENT_IDS_PER_BATCH"]), 
            margin = params["TRIPLET_MARGIN"]
        )
        return batch

# build and compile model based on specified hyperparams:
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
    use_tags = False,
    embed_tags = False,
    tag_size = None,
    tag_embedding_size = None,
    pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"],
)

model.compile(loss=loss, optimizer=params["OPTIMIZER"])

# generator re-training examples
def batch_generator():
    while True:
        batch = generate_batch(model)
        if batch[0].shape[0]:
            yield batch, np.zeros(batch[0].shape[0])

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# train model, saving results periodically
CHECKPOINT_EPOCHS = 20

total_epochs = 0
while total_epochs < params["EPOCHS"]:
    epochs_to_train = min(CHECKPOINT_EPOCHS, params["EPOCHS"] - total_epochs)
    history = model.fit(
        batch_generator(),
        epochs=epochs_to_train,
        steps_per_epoch=params["STEPS_PER_EPOCH"],
        callbacks=[epoch_tensorboard_callback]
    )

    # update total epochs
    total_epochs += epochs_to_train

    # evaluate model and save metrics:
    evaluation = evaluate(model)

    with open(experiment_dir + f"/results_{total_epochs}.json", "w") as f:
        json.dump(evaluation, f, indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")


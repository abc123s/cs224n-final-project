import json
import os
import random
from datetime import datetime
import subprocess
import math

from tensorflow import keras
import numpy as np

from preprocessing.preprocess import preprocess_train, preprocess_train_raw
from model.model import build_model
from evaluate import evaluate

# raw training examples (not yet made into triplets)
with open(os.path.join(os.path.dirname(__file__), "data/trainMatchedTrainingExamples.json")) as training_examples_data:
    raw_training_examples = json.load(training_examples_data)

# hyperparameters for re-training
ORIGINAL_EXPERIMENT_DIR = "experiments/20201126_0602_da31a2a"
RETRAINING_EPOCHS = 1
RETRAINING_EXAMPLE_TYPE = "offline_tagged_complete_with_no_match"
RETRAINING_ROUNDS = 7 # how many rounds of RETRAINING_EPOCHS to run (in series)

# load original experiment params
with open(ORIGINAL_EXPERIMENT_DIR + "/params.json", "r") as f:
    old_params = json.load(f)

# make list of params (combine old and new parameters)
params = {
    # defaults; necessary for old experiments missing these params
    "KERNEL_REGULARIZER": None,
    "REGULARIZATION_FACTOR": 0,
    "RECURRENT_REGULARIZER": None,
    "RECURRENT_REGULARIZATION_FACTOR": 0,
    "DROPOUT_RATE": 0,
    "RECURRENT_DROPOUT_RATE": 0,
    **old_params,
    "RETRAINING_EXAMPLE_TYPE": RETRAINING_EXAMPLE_TYPE,
}

if params["USE_TAGS"]:
    if params["RETRAINING_EXAMPLE_TYPE"] == "offline_tagged_complete":
        dataset, word_encoder, tag_size = preprocess_train_raw(
            raw_examples = raw_training_examples,
            triplet_options = {
                "include_no_match": False,
                "include_tags": True,
            },
            shuffle_buffer_size = params["SHUFFLE_BUFFER_SIZE"],
            batch_size = params["BATCH_SIZE"],
            shuffle_before_batch = params["SHUFFLE_BEFORE_BATCH"],
            pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
            embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
        )
    elif params["RETRAINING_EXAMPLE_TYPE"] == "offline_tagged_complete_with_no_match":
        dataset, word_encoder, tag_size = preprocess_train_raw(
            raw_examples = raw_training_examples,
            triplet_options = {
                "include_no_match": True,
                "include_tags": True,
            },
            shuffle_buffer_size = params["SHUFFLE_BUFFER_SIZE"],
            batch_size = params["BATCH_SIZE"],
            shuffle_before_batch = params["SHUFFLE_BEFORE_BATCH"],
            pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
            embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
        )
    else:
        raise ValueError(f'Selected training example type {params["RETRAINING_EXAMPLE_TYPE"]} that is not compatible with the USE_TAGS flag.')
else:
    training_example_directories = {
        "offline_full": "data/fullIngredientPhraseTripletExamples.json",
        "offline_full_v2": "data/fullIngredientPhraseTripletExamples_v2.json",
        "offline_full_complete": "data/fullIngredientPhraseTripletExamples_complete.json",
        "offline_full_complete_with_no_match": "data/fullIngredientPhraseTripletExamples_complete_include_no_match.json",
        "offline_name_only": "data/ingredientNameOnlyTripletExamples.json",
        "offline_name_only_v2": "data/ingredientNameOnlyTripletExamples_v2.json",
        "offline_name_only_complete": "data/ingredientNameOnlyTripletExamples_complete.json",
        "offline_name_only_complete_with_no_match": "data/ingredientNameOnlyTripletExamples_complete_include_no_match.json",
    }

    with open(os.path.join(os.path.dirname(__file__), training_example_directories[params["RETRAINING_EXAMPLE_TYPE"]])) as training_examples_data:
        training_examples = json.load(training_examples_data)

    dataset, word_encoder = preprocess_train(
        training_examples,
        shuffle_buffer_size = params["SHUFFLE_BUFFER_SIZE"],
        batch_size = params["BATCH_SIZE"],
        shuffle_before_batch = params["SHUFFLE_BEFORE_BATCH"],
        pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
        embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
    )
    tag_size = None

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
    use_tags = params["USE_TAGS"],
    embed_tags = params["EMBED_TAGS"],
    tag_size = tag_size,
    tag_embedding_size = params["TAG_EMBEDDING_SIZE"]
)

model.compile(loss=loss, optimizer=params["OPTIMIZER"])

# load final weights from original experiment into model:
model.load_weights(ORIGINAL_EXPERIMENT_DIR + "/model_weights")

prev_experiment_dir = ORIGINAL_EXPERIMENT_DIR 
for i in range(RETRAINING_ROUNDS):
    # make new experiment directory
    date_string = datetime.now().strftime("%Y%m%d_%H%M")
    commit_string = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    experiment_dir = "experiments/" + date_string + "_" + commit_string
    os.mkdir(experiment_dir)

    # add tensorboard logs
    epoch_tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

    # retrain model
    history = model.fit(
        dataset,
        epochs=RETRAINING_EPOCHS,
        callbacks=[epoch_tensorboard_callback]
    )

    # update params
    params["ORIGINAL_EXPERIMENT_DIR"] = prev_experiment_dir
    params["EPOCHS"] += RETRAINING_EPOCHS
    prev_experiment_dir = experiment_dir

    # save params down
    with open(experiment_dir + "/params.json", "w") as f:
        json.dump(params, f, indent=4)

    # save model weights for later usage
    model.save_weights(experiment_dir + "/model_weights")

    # evaluate model and save metrics:
    evaluation = evaluate(
        model,
        pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
        embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
    )

    with open(experiment_dir + "/results.json", "w") as f:
        json.dump(evaluation, f, indent=4)

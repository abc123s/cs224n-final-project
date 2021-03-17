import os
from datetime import datetime
import subprocess
import json

import tensorflow as tf
from tensorflow import keras

from preprocessing.preprocess import preprocess_train, preprocess_train_raw
from model.model import build_model
from evaluate import evaluate

# hyperparameters
# model structure
WORD_EMBEDDING_SIZE = 100
PRETRAINED_EMBEDDINGS = 'ing_only_ing_doc'
USE_PRETRAINED_EMBEDDING_VOCAB = True
SENTENCE_EMBEDDING_SIZE = 128
EMBEDDING_ARCHITECTURE = 'bidirectional'
USE_TAGS = True
EMBED_TAGS = True
TAG_EMBEDDING_SIZE = 4

# regularization
KERNEL_REGULARIZER = None
REGULARIZATION_FACTOR = 0
RECURRENT_REGULARIZER = None
RECURRENT_REGULARIZATION_FACTOR = 0
DROPOUT_RATE = 0
RECURRENT_DROPOUT_RATE = 0

# loss function
TRIPLET_MARGIN = 0.2

# training
OPTIMIZER = 'adam'
TRAINING_EXAMPLE_TYPE = "offline_tagged_complete_with_no_match"
SHUFFLE_BUFFER_SIZE = 200000
BATCH_SIZE = 128
SHUFFLE_BEFORE_BATCH = True
EPOCHS = 1

# raw training examples (not yet made into triplets)
with open(os.path.join(os.path.dirname(__file__), "data/trainMatchedTrainingExamples.json")) as training_examples_data:
    raw_training_examples = json.load(training_examples_data)

if USE_TAGS:
    if TRAINING_EXAMPLE_TYPE == "offline_tagged_complete":
        dataset, word_encoder, tag_size = preprocess_train_raw(
            raw_examples = raw_training_examples,
            triplet_options = {
                "include_no_match": False,
                "include_tags": True,
            },
            shuffle_buffer_size = SHUFFLE_BUFFER_SIZE,
            batch_size = BATCH_SIZE,
            shuffle_before_batch = SHUFFLE_BEFORE_BATCH,
            pretrained_embeddings = PRETRAINED_EMBEDDINGS if USE_PRETRAINED_EMBEDDING_VOCAB else None,
            embedding_size = WORD_EMBEDDING_SIZE if USE_PRETRAINED_EMBEDDING_VOCAB else None
        )
    elif TRAINING_EXAMPLE_TYPE == "offline_tagged_complete_with_no_match":
        dataset, word_encoder, tag_size = preprocess_train_raw(
            raw_examples = raw_training_examples,
            triplet_options = {
                "include_no_match": True,
                "include_tags": True,
            },
            shuffle_buffer_size = SHUFFLE_BUFFER_SIZE,
            batch_size = BATCH_SIZE,
            shuffle_before_batch = SHUFFLE_BEFORE_BATCH,
            pretrained_embeddings = PRETRAINED_EMBEDDINGS if USE_PRETRAINED_EMBEDDING_VOCAB else None,
            embedding_size = WORD_EMBEDDING_SIZE if USE_PRETRAINED_EMBEDDING_VOCAB else None
        )
    else:
        raise ValueError(f'Selected training example type {TRAINING_EXAMPLE_TYPE} that is not compatible with the USE_TAGS flag.')
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

    with open(os.path.join(os.path.dirname(__file__), training_example_directories[TRAINING_EXAMPLE_TYPE])) as training_examples_data:
        training_examples = json.load(training_examples_data)    

    dataset, word_encoder = preprocess_train(
        training_examples,
        shuffle_buffer_size = SHUFFLE_BUFFER_SIZE,
        batch_size = BATCH_SIZE,
        shuffle_before_batch = SHUFFLE_BEFORE_BATCH,
        pretrained_embeddings = PRETRAINED_EMBEDDINGS if USE_PRETRAINED_EMBEDDING_VOCAB else None,
        embedding_size = WORD_EMBEDDING_SIZE if USE_PRETRAINED_EMBEDDING_VOCAB else None
    )
    tag_size = None

# build model
model, loss = build_model(
    word_encoder = word_encoder,
    word_embedding_size = WORD_EMBEDDING_SIZE,
    sentence_embedding_size = SENTENCE_EMBEDDING_SIZE,
    embedding_architecture = EMBEDDING_ARCHITECTURE,
    triplet_margin = TRIPLET_MARGIN,
    kernel_regularizer = KERNEL_REGULARIZER,
    kernel_regularization_factor = REGULARIZATION_FACTOR,
    recurrent_regularizer = RECURRENT_REGULARIZER,
    recurrent_regularization_factor = RECURRENT_REGULARIZATION_FACTOR,
    dropout_rate = DROPOUT_RATE,
    recurrent_dropout_rate = RECURRENT_DROPOUT_RATE,
    use_tags = USE_TAGS,
    embed_tags = EMBED_TAGS,
    tag_size = tag_size,
    tag_embedding_size = TAG_EMBEDDING_SIZE,
    pretrained_embeddings = PRETRAINED_EMBEDDINGS
)

model.summary()

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

# compile model
model.compile(loss=loss, optimizer=OPTIMIZER)
history = model.fit(dataset,
                    epochs=EPOCHS,
                    callbacks=[epoch_tensorboard_callback])

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(
        {
            "WORD_EMBEDDING_SIZE": WORD_EMBEDDING_SIZE,
            "PRETRAINED_EMBEDDINGS": PRETRAINED_EMBEDDINGS,
            "USE_PRETRAINED_EMBEDDING_VOCAB": USE_PRETRAINED_EMBEDDING_VOCAB,
            "SENTENCE_EMBEDDING_SIZE": SENTENCE_EMBEDDING_SIZE,
            "EMBEDDING_ARCHITECTURE": EMBEDDING_ARCHITECTURE,
            "USE_TAGS": USE_TAGS,
            "EMBED_TAGS": EMBED_TAGS,
            "TAG_EMBEDDING_SIZE": TAG_EMBEDDING_SIZE,
            "KERNEL_REGULARIZER": KERNEL_REGULARIZER,
            "REGULARIZATION_FACTOR": REGULARIZATION_FACTOR,
            "RECURRENT_REGULARIZER": RECURRENT_REGULARIZER,
            "RECURRENT_REGULARIZATION_FACTOR": RECURRENT_REGULARIZATION_FACTOR,
            "DROPOUT_RATE": DROPOUT_RATE,
            "RECURRENT_DROPOUT_RATE": RECURRENT_DROPOUT_RATE,
            "TRIPLET_MARGIN": TRIPLET_MARGIN,
            "OPTIMIZER": OPTIMIZER,
            "TRAINING_EXAMPLE_TYPE": TRAINING_EXAMPLE_TYPE,
            "SHUFFLE_BUFFER_SIZE": SHUFFLE_BUFFER_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "SHUFFLE_BEFORE_BATCH": SHUFFLE_BEFORE_BATCH,
            "EPOCHS": EPOCHS,
        },
        f,
        indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(
    model,
    pretrained_embeddings = PRETRAINED_EMBEDDINGS if USE_PRETRAINED_EMBEDDING_VOCAB else None,
    embedding_size = WORD_EMBEDDING_SIZE if USE_PRETRAINED_EMBEDDING_VOCAB else None
)

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent=4)

import json
import os

import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer, TagTokenizer
from model.model import build_model
from evaluate import evaluate

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

# evaluate model and save metrics:
evaluation = evaluate(model)

with open(experiment_dir + "/results_redo.json", "w") as f:
    json.dump(evaluation, f, indent=4)

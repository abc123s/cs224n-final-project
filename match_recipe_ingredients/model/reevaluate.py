import json
import os

import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer, TagTokenizer
from preprocessing.preprocess import create_word_encoder
from model.model import build_model
from evaluate import evaluate

ingredientPhraseTokenizer = IngredientPhraseTokenizer()
tagTokenizer = TagTokenizer()
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder


# create tag_encoder (to get tag vocab size)
with open(os.path.join(os.path.dirname(__file__), "preprocessing/tag_list.json")) as tag_list_data:
    tag_list = json.load(tag_list_data)

tag_encoder = TokenTextEncoder(tag_list, oov_buckets=0, tokenizer=tagTokenizer)

experiment_dir = "experiments/20201203_1651_63cfe44"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# create word_encoder (to get vocab size)
word_encoder = create_word_encoder(
    pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
    embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
)

# build and compile model based on experiment params:
model, _ = build_model(
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
    tag_size = tag_encoder.vocab_size,
    tag_embedding_size = params["TAG_EMBEDDING_SIZE"]
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(
    model,
    pretrained_embeddings = params["PRETRAINED_EMBEDDINGS"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None,
    embedding_size = params["WORD_EMBEDDING_SIZE"] if params["USE_PRETRAINED_EMBEDDING_VOCAB"] else None
)

with open(experiment_dir + "/results_redo.json", "w") as f:
    json.dump(evaluation, f, indent=4)

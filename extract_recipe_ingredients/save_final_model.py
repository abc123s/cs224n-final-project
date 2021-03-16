import json

import tensorflow as tf
from tensorflow import keras

from preprocess_simple import preprocess as preprocess_simple, load_examples as load_simple_examples
from preprocess_original import preprocess as preprocess_original, crfFile2Examples
from preprocess_manual import preprocess as preprocess_manual, load_examples as load_manual_examples
from preprocess_combined import preprocess as preprocess_combined
from build_model import build_model
from masked_accuracy import SparseCategoricalAccuracyMaskZeros
from tokenizer import IngredientPhraseTokenizer

ingredientPhraseTokenizer = IngredientPhraseTokenizer()

# modify this line to change which experiment directory you wish
# save a full model for
experiment_dir = "experiments/20200819_2356_f360e93"

preprocessors = {
    'simple': preprocess_simple,
    'original': preprocess_original,
    'manual': preprocess_manual,
    'combined': preprocess_combined,
}


# used when running error analysis on a fine-tuned model
def load_original():
    return crfFile2Examples("./data/train.crf")


def load_simple():
    return load_simple_examples("./data", "train")


def load_combined():
    nyt_train_examples = load_simple_examples("./data", "train")

    manual_train_examples = load_manual_examples("./data",
                                                 "manually_tagged_train")
    return [
        *nyt_train_examples,
        *manual_train_examples,
    ]


def load_manual():
    return load_manual_examples("./data", "train")


example_loader = {
    'original': load_original,
    'simple': load_simple,
    'manual': load_manual,
    'combined': load_combined,
}

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# grab dev set to do error analysis on, making sure to properly handle
# fine-tuned models (which use two different datasets)
vocab_examples = example_loader[params.get("PREPROCESSOR", "original")]()
if params.get("ORIGINAL_EXPERIMENT_DIR", None):
    _, dev_data, _, _, _, word_encoder, tag_encoder = preprocess_manual(
        "./data", vocab_examples)
else:
    preprocess = preprocessors[params.get("PREPROCESSOR", 'original')]
    _, dev_data, _, _, _, word_encoder, tag_encoder = preprocess("./data")

# build and compile model based on experiment params:
model = build_model(
    architecture=params["ARCHITECTURE"],
    embedding_units=params["EMBEDDING_UNITS"],
    num_recurrent_layers=params.get("NUM_RECURRENT_LAYERS", 1),
    recurrent_units=params["RECURRENT_UNITS"],
    regularizer=params.get("REGULARIZER", None),
    regularization_factor=params.get("REGULARIZATION_FACTOR", 0),
    dropout_rate=params.get("DROPOUT_RATE", 0),
    recurrent_dropout_rate=params["RECURRENT_DROPOUT_RATE"],
    vocab_size=word_encoder.vocab_size,
    tag_size=tag_encoder.vocab_size,
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# save down full model, not just weights
model.save('extract_recipe_ingredients/model')

# save down vocab list and tag list (necessary for creating the
# tokenizer, encoder and decoder)
vocab_list = sorted(
    set([
        word for example in vocab_examples
        for word in ingredientPhraseTokenizer.tokenize(example[0])
    ]))

with open('extract_recipe_ingredients/vocab_list.json', 'w') as outfile:
    json.dump(vocab_list, outfile)

tag_list = sorted(
    set([tag for example in vocab_examples for tag in example[1]]))

with open('extract_recipe_ingredients/tag_list.json', 'w') as outfile:
    json.dump(tag_list, outfile)

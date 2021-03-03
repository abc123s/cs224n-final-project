import datetime
import json

import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
TokenTextEncoder = tfds.features.text.TokenTextEncoder
Tokenizer = tfds.features.text.Tokenizer

from tokenizer import IngredientPhraseTokenizer, TagTokenizer
ingredientPhraseTokenizer = IngredientPhraseTokenizer()
tagTokenizer = TagTokenizer()

test_examples = [
    "8 slices of crusty, whole grain bread",
    "300 g / 4 cups (packed cauliflower small florets (1 small / 1/2 large cauliflower))",
    "1 celery stalk, (chopped)",
    "5 tablespoons mayonnaise",
    "1 (10-oz.) French bread loaf",
    "2 green bell peppers, chopped",
    "Chopped fresh parsley, for serving",
    "1 tablespoon Chinkiang vinegar (use a not-too-fancy balsamic vinegar in its place if unavailable)",
    "2 tbsp Thai red curry paste",
    "1/3 cup red enchilada sauce",
]
start_loading_encoders = datetime.datetime.now()
with open("extract_recipe_ingredients/vocab_list.json") as vocab_list_data:
    vocab_list = json.load(vocab_list_data)

with open("extract_recipe_ingredients/tag_list.json") as tag_list_data:
    tag_list = json.load(tag_list_data)

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)
tag_encoder = TokenTextEncoder(tag_list, oov_buckets=0, tokenizer=tagTokenizer)

print(
    f'Loading encoders took {datetime.datetime.now() - start_loading_encoders}.'
)

start_creating_dataset = datetime.datetime.now()


def example_generator():
    for example in test_examples:
        yield word_encoder.encode(example)


test_dataset = tf.data.Dataset.from_generator(example_generator,
                                              output_types=tf.int32)
test_batches = test_dataset.take(len(test_examples)).padded_batch(
    len(test_examples), padded_shapes=[None])

print(
    f'Creating dataset took {datetime.datetime.now() - start_creating_dataset}.'
)

start_loading_model = datetime.datetime.now()
model = keras.models.load_model("extract_recipe_ingredients/model",
                                compile=False)

print(f'Loading model took {datetime.datetime.now() - start_loading_model}.')

start_predicting = datetime.datetime.now()
for sentences in test_batches:
    model_outputs = model(sentences)
    for model_output, sentence in zip(model_outputs, sentences):
        # get clean python list with sentence tokens (rather than tensor)
        clean_sentence = sentence.numpy().tolist()

        # make prediction by selecting most likely tag for each token
        prediction = keras.backend.flatten(
            keras.backend.argmax(model_output)).numpy().tolist()

        # trim prediction
        padding_start = clean_sentence.index(
            0) if clean_sentence[-1] == 0 else len(clean_sentence)
        trimmed_prediction = prediction[0:padding_start]

        # decode sentence and labels, and check if something is wrong with padding
        decoded_sentence = word_encoder.decode(sentence).split(' ')
        decoded_prediction = tag_encoder.decode(trimmed_prediction).split(' ')

        # print results
        print(decoded_sentence)
        print(decoded_prediction)
        print()

print(
    f'Making {len(test_examples)} predictions took {datetime.datetime.now() - start_predicting}.'
)

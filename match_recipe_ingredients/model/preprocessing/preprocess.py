import json
import os
import random

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

from preprocessing.tokenizer import IngredientPhraseTokenizer, TagTokenizer
ingredientPhraseTokenizer = IngredientPhraseTokenizer()
tagTokenizer = TagTokenizer()


# ingredient dictionary (used to generate triplets from raw examples)
with open(os.path.join(os.path.dirname(__file__), "../data/ingredientDictionary.json")) as ingredient_dictionary_data:
    ingredient_dictionary = json.load(ingredient_dictionary_data)

# create word_encoder
with open(os.path.join(os.path.dirname(__file__), "vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

# create tag_encoder
with open(os.path.join(os.path.dirname(__file__), "tag_list.json")) as tag_list_data:
    tag_list = json.load(tag_list_data)

tag_encoder = TokenTextEncoder(tag_list, oov_buckets=0, tokenizer=tagTokenizer)


TRUNCATE_LENGTH = 50
# truncate all examples beyond 50 tokens, pad short
# examples up to 50 tokens
def truncate_or_pad(example):
    if len(example) > TRUNCATE_LENGTH:
        return example[0:TRUNCATE_LENGTH]
    else:
        return example + [0] * (TRUNCATE_LENGTH - len(example))

# convert triplet training examples into batched tf Dataset
def preprocess_train(all_examples, shuffle_buffer_size, batch_size, shuffle_before_batch):
    # exclude all examples with empty strings
    # necessary for ingredient name only examples, because there could be tagging mistakes that
    # cause there to be no ingredient name
    examples = [example for example in all_examples if len(example[0]) and len(example[1]) and len(example[2])]
    random.shuffle(examples)

    def example_generator():
        for anchor, positive, negative in examples:
            encoded_anchor = word_encoder.encode(anchor)
            encoded_positive = word_encoder.encode(positive)
            encoded_negative = word_encoder.encode(negative)
            yield (
                truncate_or_pad(encoded_anchor),
                truncate_or_pad(encoded_positive),
                truncate_or_pad(encoded_negative)
            ), 0


    test_dataset = tf.data.Dataset.from_generator(example_generator,
                                                  output_types=((tf.int32, tf.int32, tf.int32), tf.int32))

    if shuffle_before_batch:
        return test_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).batch(batch_size), word_encoder.vocab_size
    else:
        return test_dataset.batch(batch_size).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True), word_encoder.vocab_size

def encode_tags(tags, encoded_text):
    if tags == None:
        tags = ['B-NAME', *(["I-NAME"] * (len(encoded_text) - 1))]

    return [tag_encoder.encode(tag)[0] for tag in tags]

def preprocess_train_raw(raw_examples, triplet_options, shuffle_buffer_size, batch_size, shuffle_before_batch):
    triplets = []
    for raw_example in raw_examples:
        example_text = raw_example["original"]
        example_ingredient_id = (
            raw_example["ingredients"][0]["ingredient"] and 
            raw_example["ingredients"][0]["ingredient"]["id"]
        )
        example_tags = raw_example["tags"]

        if example_ingredient_id != None:
            matching_dictionary_entry = ingredient_dictionary[example_ingredient_id]
            if triplet_options["include_tags"]:
                for other_ingredient_id, other_entry in ingredient_dictionary.items():
                    if other_ingredient_id != example_ingredient_id:
                        triplets.append([
                            [example_text, example_tags],
                            [matching_dictionary_entry, None],
                            [other_entry, None]
                        ])
            else:
                for other_ingredient_id, other_entry in ingredient_dictionary.items():
                    if other_ingredient_id != exampleIngredient_id:
                        triplets.append([
                            example_text,
                            matching_dictionary_entry,
                            other_entry,
                        ])
        elif triplet_options["include_no_match"]:
            if triplet_options["include_tags"]:
                for _, entry in ingredient_dictionary.items():
                    triplets.append([
                        [example_text, example_tags],
                        [example_text, example_tags],
                        [entry, None]
                    ])
            else:
                for _, entry in ingredient_dictionary.items():
                    triplets.append([
                        example_text,
                        example_text,
                        entry,
                    ])

    random.shuffle(triplets)

    if triplet_options["include_tags"]:
        def example_generator():
            for (anchor, anchor_tags), (positive, positive_tags), (negative, negative_tags) in triplets:
                encoded_anchor = word_encoder.encode(anchor)
                encoded_anchor_tags = encode_tags(anchor_tags, encoded_anchor)
                encoded_positive = word_encoder.encode(positive)
                encoded_positive_tags = encode_tags(positive_tags, encoded_positive)
                encoded_negative = word_encoder.encode(negative)
                encoded_negative_tags = encode_tags(negative_tags, encoded_negative)
                
                yield (
                    truncate_or_pad(encoded_anchor),
                    truncate_or_pad(encoded_anchor_tags), 
                    truncate_or_pad(encoded_positive),
                    truncate_or_pad(encoded_positive_tags),
                    truncate_or_pad(encoded_negative),
                    truncate_or_pad(encoded_negative_tags)
                ), 0

        test_dataset = tf.data.Dataset.from_generator(example_generator,
                                                    output_types=((tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32), tf.int32))

        if shuffle_before_batch:
            return test_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).batch(batch_size), word_encoder.vocab_size, tag_encoder.vocab_size
        else:
            return test_dataset.batch(batch_size).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True), word_encoder.vocab_size, tag_encoder.vocab_size
    else:
        def example_generator():
            for anchor, positive, negative in examples:
                encoded_anchor = word_encoder.encode(anchor)
                encoded_positive = word_encoder.encode(positive)
                encoded_negative = word_encoder.encode(negative)
                yield (
                    truncate_or_pad(encoded_anchor),
                    truncate_or_pad(encoded_positive),
                    truncate_or_pad(encoded_negative)
                ), 0

        test_dataset = tf.data.Dataset.from_generator(example_generator,
                                                    output_types=((tf.int32, tf.int32, tf.int32), tf.int32))

        if shuffle_before_batch:
            return test_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).batch(batch_size), word_encoder.vocab_size, None
        else:
            return test_dataset.batch(batch_size).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True), word_encoder.vocab_size, None



# convert batch of manually selected training examples
# into format accepted by model.fit
def preprocess_train_batch(example_batch):
    processed_anchors = []
    processed_positives = []
    processed_negatives = []
    for anchor, positive, negative in example_batch:
        processed_anchors.append(
            truncate_or_pad(word_encoder.encode(anchor))
        )
        processed_positives.append(
            truncate_or_pad(word_encoder.encode(positive))
        )
        processed_negatives.append(
            truncate_or_pad(word_encoder.encode(negative))
        )

    return [np.array(processed_anchors), np.array(processed_positives), np.array(processed_negatives)]

# convert list of raw test examples into format
# that embedding predict on 
def preprocess_test(examples):
    use_tags = isinstance(examples[0], list)
    if use_tags:
        encoded_example_text = []
        encoded_example_tags = []
        for text, tags in examples:
            encoded_text = word_encoder.encode(text)
            encoded_tags = encode_tags(tags, encoded_text)
            encoded_example_text.append(truncate_or_pad(encoded_text))
            encoded_example_tags.append(truncate_or_pad(encoded_tags))

        return [
            tf.constant(encoded_example_text),
            tf.constant(encoded_example_tags),
        ]
    else:
        encoded_examples = [truncate_or_pad(word_encoder.encode(example)) for example in examples]

        return tf.constant(encoded_examples)
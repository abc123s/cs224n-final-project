import tensorflow as tf
from tensorflow import keras

def construct_regularizer(regularizer, regularization_factor):
    # construct regularizer
    if regularizer == 'l2':
        regularizer = keras.regularizers.l2(regularization_factor)
    elif regularizer == 'l1':
        regularizer = keras.regularizers.l1(regularization_factor)
    elif regularizer == 'l1_l2':
        regularizer = keras.regularizers.l1_l2(regularization_factor)
    else:
        regularizer = None

    return regularizer


def build_embedding(
    vocab_size,
    word_embedding_size,
    sentence_embedding_size,
    architecture,
    kernel_regularizer,
    kernel_regularization_factor,
    recurrent_regularizer,
    recurrent_regularization_factor,
    dropout_rate,
    recurrent_dropout_rate,
    use_tags,
    embed_tags,
    tag_size,
    tag_embedding_size
):
    kernel_regularizer_instance = construct_regularizer(
        kernel_regularizer,
        kernel_regularization_factor
    )

    recurrent_regularizer_instance = construct_regularizer(
        recurrent_regularizer,
        recurrent_regularization_factor
    )

    if use_tags:
        tokens = keras.Input(shape=(50,), name = 'tokens')
        tags = keras.Input(shape=(50,), name = 'tags')

        token_embedding_layer = keras.layers.Embedding(vocab_size, word_embedding_size, mask_zero = True, name = "token_embedding")
        embedded_tokens = token_embedding_layer(tokens)

        if embed_tags:
            tag_embedding_layer = keras.layers.Embedding(tag_size, tag_embedding_size, name = 'tag_embedding')
            embedded_tags = tag_embedding_layer(tags)
        else:
            embedded_tags = tf.expand_dims(tags, axis = -1, name = 'tag_expand_dim')

        combined_input = keras.layers.concatenate(
            [embedded_tokens, embedded_tags],
            axis = 2,
            name = 'concat_tokens_tags'
        )

        if architecture == 'simple':
            recurrent_layer = keras.layers.LSTM(
                sentence_embedding_size,
                kernel_regularizer = kernel_regularizer_instance,
                recurrent_regularizer = recurrent_regularizer_instance,
                dropout = dropout_rate,
                recurrent_dropout = recurrent_dropout_rate,
            )
        elif architecture == 'bidirectional':
            recurrent_layer = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    sentence_embedding_size,
                    kernel_regularizer = kernel_regularizer_instance,
                    recurrent_regularizer = recurrent_regularizer_instance,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                ),
                merge_mode = 'ave'
            )

        output = recurrent_layer(combined_input)

        return keras.Model(inputs = [tokens, tags], outputs = output, name='embedding')

    else:
        if architecture == 'simple':
            return keras.Sequential(
                [
                    keras.layers.Embedding(vocab_size, word_embedding_size, mask_zero = True),
                    keras.layers.LSTM(
                        sentence_embedding_size,
                        kernel_regularizer = kernel_regularizer_instance,
                        recurrent_regularizer = recurrent_regularizer_instance,
                        dropout = dropout_rate,
                        recurrent_dropout = recurrent_dropout_rate,
                    )
                ],
                name = 'embedding'
            )
        if architecture == 'bidirectional':
            return keras.Sequential(
                [
                    keras.layers.Embedding(vocab_size, word_embedding_size, mask_zero = True),
                    keras.layers.Bidirectional(
                        keras.layers.LSTM(
                            sentence_embedding_size,
                            kernel_regularizer = kernel_regularizer_instance,
                            recurrent_regularizer = recurrent_regularizer_instance,
                            dropout = dropout_rate,
                            recurrent_dropout = recurrent_dropout_rate,
                        ),
                        merge_mode = 'ave'
                    ),
                ],
                name = 'embedding'
            )

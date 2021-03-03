import tensorflow as tf
from tensorflow import keras

from model.embedding import build_embedding 

def build_model(
    vocab_size,
    word_embedding_size,
    sentence_embedding_size,
    embedding_architecture,
    triplet_margin,
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
    # construct model
    embedding = build_embedding(
        vocab_size = vocab_size,
        word_embedding_size = word_embedding_size,
        sentence_embedding_size = sentence_embedding_size,
        architecture = embedding_architecture,
        kernel_regularizer = kernel_regularizer,
        kernel_regularization_factor = kernel_regularization_factor,
        recurrent_regularizer = recurrent_regularizer,
        recurrent_regularization_factor = recurrent_regularization_factor,
        dropout_rate = dropout_rate,
        recurrent_dropout_rate = recurrent_dropout_rate,
        use_tags = use_tags,
        embed_tags = embed_tags,
        tag_size = tag_size,
        tag_embedding_size = tag_embedding_size
    )

    if use_tags:
        anchor_tokens = keras.layers.Input(shape=(50,), name = 'anchor_tokens')
        anchor_tags = keras.layers.Input(shape=(50,), name = 'anchor_tags')
        positive_tokens = keras.layers.Input(shape=(50,), name = 'positive_tokens')
        positive_tags = keras.layers.Input(shape=(50,), name = 'positive_tags')
        negative_tokens = keras.layers.Input(shape=(50,), name = 'negative_tokens')
        negative_tags = keras.layers.Input(shape=(50,), name = 'negative_tags')
        inputs = [
            anchor_tokens,
            anchor_tags,
            positive_tokens,
            positive_tags,
            negative_tokens,
            negative_tags,
        ]

        anchor_embedding = embedding([anchor_tokens, anchor_tags])
        positive_embedding = embedding([positive_tokens, positive_tags])
        negative_embedding = embedding([negative_tokens, negative_tags])
    else:
        anchor = keras.layers.Input(shape=(50,), name = 'anchor')
        positive = keras.layers.Input(shape=(50,), name = 'positive')
        negative = keras.layers.Input(shape=(50,), name = 'negative')
        inputs = [
            anchor,
            positive,
            negative,
        ]

        anchor_embedding = embedding(anchor)
        positive_embedding = embedding(positive)
        negative_embedding = embedding(negative)

    output = keras.layers.concatenate(
        [anchor_embedding, positive_embedding, negative_embedding],
        axis = 1,
        name = 'concat_anchor_positive_negative'
    )

    model = keras.models.Model(inputs, output)

    # construct triplet loss function
    def triplet_loss(_, y_pred):
        # extract embeddings for anchor, positive, and negative examples
        anchor_embeddings = y_pred[:,:sentence_embedding_size]
        positive_embeddings = y_pred[:,sentence_embedding_size:2*sentence_embedding_size]
        negative_embeddings = y_pred[:,2*sentence_embedding_size:]

        # compute (mean) distance between anchors and positives and anchors and negatives
        positive_dist = tf.math.reduce_mean(tf.square(anchor_embeddings - positive_embeddings), axis=1)
        negative_dist = tf.math.reduce_mean(tf.square(anchor_embeddings - negative_embeddings), axis=1)

        # compute triplet loss with specified margin
        return tf.math.maximum(positive_dist - negative_dist + triplet_margin, 0.)

    return model, triplet_loss



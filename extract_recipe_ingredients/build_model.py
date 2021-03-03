import numpy as np
from tensorflow import keras

def build_recurrent_layers(
    architecture,
    num_layers,
    units,
    regularizer,
    regularization_factor,
    dropout_rate,
    recurrent_dropout_rate
):
    # construct regularizer
    if regularizer == 'l2':
        regularizer = keras.regularizers.l2(regularization_factor)
    elif regularizer == 'l1':
        regularizer = keras.regularizers.l1(regularization_factor)
    elif regularizer == 'l1_l2':
        regularizer = keras.regularizers.l1_l2(regularization_factor)
    else:
        regularizer = None

    # construct model layers
    layers = []
    if architecture == 'rnn':
        for _ in range(num_layers):
            layers.append(
                keras.layers.SimpleRNN(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'brnn':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.SimpleRNN(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )

    if architecture == 'gru':
        for _ in range(num_layers):
            layers.append(
                keras.layers.GRU(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'bgru':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.GRU(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )

    if architecture == 'lstm':
        for _ in range(num_layers):
            layers.append(
                keras.layers.LSTM(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                )
            )

    if architecture == 'blstm':
        for _ in range(num_layers):
            layers.append(
                keras.layers.Bidirectional(keras.layers.LSTM(
                    units,
                    kernel_regularizer = regularizer,
                    recurrent_regularizer = regularizer,
                    dropout = dropout_rate,
                    recurrent_dropout = recurrent_dropout_rate,
                    return_sequences = True
                ))
            )
    
    return layers

# helper function that reads a file of pre-trained embeddings into
# a lookup dict { [word]: embedding vector }
EMBEDDING_PATHS = {
    'glove': {
        '50': '../pretrained_embeddings/glove.6B.50d.txt',
        '100': '../pretrained_embeddings/glove.6B.100d.txt',
        '200': '../pretrained_embeddings/glove.6B.200d.txt',
        '300': '../pretrained_embeddings/glove.6B.300d.txt',
    }
}
def load_pretrained_embedding(pretrained_embeddings, embedding_units):
    embedding_type_paths = EMBEDDING_PATHS.get(pretrained_embeddings)
    if embedding_type_paths is None:
        raise Exception(f'Could not find pretrained embeddings of type {pretrained_embeddings}')

    embedding_path = embedding_type_paths.get(str(embedding_units))
    if embedding_path is None:
        raise Exception(f'Could not find {pretrained_embeddings} embeddings with dimension {embedding_units}')

    word_to_embedding = {}
    with open(embedding_path) as f:
        for line in f:
            word, embedding = line.split(maxsplit=1)
            embedding = np.fromstring(embedding, "f", sep = " ")
            word_to_embedding[word] = embedding

    print(f'Loaded {pretrained_embeddings} embeddings ({len(word_to_embedding)} embeddings)')

    return word_to_embedding

# helper function to construct embedding matrix to initialize
# embedding layer with
def construct_embedding_matrix(embedding_units, word_encoder, word_to_embedding):
    # initialize embedding_matrix
    embedding_matrix = np.zeros((word_encoder.vocab_size, embedding_units))
    
    # iterate through all words in vocab, ignoring padding (0) and UNK token,
    # and fill in embedding matrix
    filled = 0
    for i in range(1, word_encoder.vocab_size - 1):
        word = word_encoder.decode([i])
        embedding = word_to_embedding.get(word)
        if embedding is not None:
            embedding_matrix[i] = embedding
            filled += 1

    print(f'Filled embedding matrix with {filled} pretrained embeddings; {word_encoder.vocab_size - filled} vocab words missing (including PAD and UNK).')

    return embedding_matrix

# helper function to construct embedding layer of model
def build_embedding_layer(embedding_units, word_encoder, pretrained_embeddings = None):
    # construct non-pretrained embedding layer
    if pretrained_embeddings == None:
        return keras.layers.Embedding(word_encoder.vocab_size, embedding_units, mask_zero = True)
    
    # construct pretrained embedding layer
    else:
        # load in pretrained embeddings
        word_to_embedding = load_pretrained_embedding(pretrained_embeddings, embedding_units)
        
        # fill embedding_matrix with pretrained embedding values
        embedding_matrix = construct_embedding_matrix(embedding_units, word_encoder, word_to_embedding)

        # construct embedding layer
        return keras.layers.Embedding(
            word_encoder.vocab_size,
            embedding_units,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            mask_zero = True
        )


def build_model(
    architecture,
    embedding_units,
    num_recurrent_layers,
    recurrent_units,
    regularizer,
    regularization_factor,
    dropout_rate,
    recurrent_dropout_rate,
    word_encoder,
    tag_size,
    pretrained_embeddings = None
):
    embedding_layer = build_embedding_layer(
        embedding_units,
        word_encoder,
        pretrained_embeddings
    )

    recurrent_layers = build_recurrent_layers(
        architecture,
        num_recurrent_layers,
        recurrent_units,
        regularizer,
        regularization_factor,
        dropout_rate,
        recurrent_dropout_rate,
    )

    return keras.Sequential([
        embedding_layer,
        *recurrent_layers,
        keras.layers.Dense(tag_size),
    ])
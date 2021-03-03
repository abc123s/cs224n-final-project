# adapted from problem set 2
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces, i.e. use split(' ').
    For normalization, you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split(' ')
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counts = {}
    dictionary = {}
    index = 0

    for message in messages:
        words = get_words(message)
        for word in words:
            counts[word] = counts[word] + 1 if word in counts else 1

            # only add words to dictionary if it occurs at least 5 times
            # (and don't re-add word if we've already added it once)
            if counts[word] == 5:
                dictionary[word] = index
                index += 1

    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # initialize the output numpy array (zero occurrences of
    # every word in each message)
    transformed = np.zeros((len(messages), len(word_dictionary)))

    # fill in numpy array by counting occurrences
    # of words in message
    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            # ignore words not in dictionary
            # (rare words)
            if word in word_dictionary:
                j = word_dictionary[word]
                transformed[i][j] += 1


    return transformed
    # *** END CODE HERE ***

def main():
    with open('./naiveBayesTrain.json') as train_file:
        train_data = json.load(train_file)

    with open('./naiveBayesDev.json') as test_file:
        test_data = json.load(test_file)

    train_messages = []
    train_labels = []
    for example in train_data:
        train_messages.append(example['text'])
        train_labels.append(int(example['label']))

    test_messages = []
    test_labels = []
    for example in test_data:
        test_messages.append(example['text'])
        test_labels.append(int(example['label']))


    dictionary = create_dictionary(train_messages)
    train_matrix = transform_text(train_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    clf = MultinomialNB()
    clf.fit(train_matrix, train_labels)
    
    train_predicts = clf.predict(train_matrix)
    train_correct = np.sum(train_predicts == train_labels)
    train_total = len(train_predicts)
    print('train accuracy: ', train_correct / train_total)

    test_predicts = clf.predict(test_matrix)
    test_correct = np.sum(test_predicts == test_labels)
    test_total = len(test_predicts)
    print('test accuracy: ', test_correct / test_total)


if __name__ == "__main__":
    main()

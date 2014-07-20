"""
Data loading utilities
"""

import gzip
import cPickle
import numpy
from scipy.io import loadmat


# TODO: pickle datasets
def mnist():
    """ load mnist data from ? """

    f = gzip.open("data/mnist.pkl.gz")
    data = cPickle.load(f)
    f.close()

    return data


def kaggle_decoding():
    """ load kaggle decoding data from ? """

    train_subject_range = range(1,2)
    valid_subject_range = range(11,12)
    test_subject_range = range(17,18)

    train_x = []
    train_y = []

    for subject in train_subject_range:
        filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        train_x.append(data['X'])
        train_y.append(data['y'])

    train_x = numpy.vstack(train_x)
    train_y = numpy.concatenate(train_y)

    valid_x = []
    valid_y = []

    for subject in valid_subject_range:
        filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        valid_x.append(data['X'])
        valid_y.append(data['y'])

    valid_x = numpy.vstack(valid_x)
    valid_x = numpy.concatenate(valid_y)

    test_x = []
    test_y = []

    for subject in test_subject_range:
        filename = 'data/kaggle_decoding/test_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        test_x.append(data['X'])
        test_y.append(data['Id'])

    test_x = numpy.vstack(test_x)
    test_y = numpy.concatenate(test_y)

    return ((train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y))


def reshape(data):
    """ make x values 2d and vectorize labels """

    train_x, train_y = data[0]
    valid_x, valid_y = data[1]
    test_x, test_y = data[2]

    train_x = reshape_examples(train_x)
    valid_x = reshape_examples(valid_x)
    test_x = reshape_examples(test_x)

    train_y = vectorize_labels(train_y)
    valid_y = vectorize_labels(valid_y)

    return ((train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y))


def reshape_examples(set_x):
    """ convert examples from (n,) to (n,1) """

    rows = set_x.shape[0]
    cols = set_x.shape[1]
    return set_x.reshape((rows, cols, 1))


def vectorize_labels(set_y):
    """ convert labels from n to [0,0,n=1,0]  """

    rows = len(set_y)
    cols = numpy.max(set_y) + 1
    vectorized_set_y = numpy.zeros(shape=(rows, cols))

    for vector, y in zip(vectorized_set_y, set_y):
        vector[y] = 1

    return vectorized_set_y.reshape((rows, cols, 1))

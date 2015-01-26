""" Load datasets
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from os.path import dirname
from os.path import join

import gzip
import cPickle


class Dataset(object):
    """"""
    def __init__(self, train, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test


class Data(object):
    """"""
    def __init__(self, X, y=None):
        self.samples, self.features = X.shape
        self.X = X

        if y is not None:
            self.classes = len(np.unique(y))
            self.y = y

    def __eq__(self, other):
        return np.all(self.X == other.X)

    def __ne__(self, other):
        return not self == other

    @property
    def is_supervised(self):
        return self.y

    def batches(self, batch_size):
        return self.samples / batch_size


def load_mnist():
    """Load and return the mnist digit dataset (classification).

    :reference:

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    module_path = dirname(__file__)
    with gzip.open(join(module_path, 'mnist', 'mnist.pkl.gz')) as data_file:
        return cPickle.load(data_file)


def load_cifar_10():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError


def load_cifar_100():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError


def load_svhn():
    """Load and return the mnist digit dataset (classification).

    Each datapoint is a 28x28 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class            ~7000
    Samples total                70000
    Dimensionality                 784
    Features                floats 0-1
    =================   ==============

    """
    raise NotImplementedError

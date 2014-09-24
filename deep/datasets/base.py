"""
Base IO code for all datasets
"""

from os.path import dirname
from os.path import join

import gzip
import cPickle


def load_mnist():
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

    module_path = dirname(__file__)

    with gzip.open(join(module_path, 'data', 'mnist.pkl.gz')) as data_file:
        data_file = cPickle.load(data_file)
    return data_file
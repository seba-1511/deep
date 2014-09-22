"""
Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import csv
from os.path import dirname
from os.path import join

import numpy as np
import gzip
import cPickle


def load_iris():
    """Load and return the iris dataset (classification).

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    """

    module_path = dirname(__file__)
    with open(join(module_path, 'data', 'iris.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target


def load_digits():
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    """

    module_path = dirname(__file__)
    data_file = np.loadtxt(join(module_path, 'data', 'digits.csv.gz'),
                           delimiter=',')
    data = data_file[:, :-1]
    target = data_file[:, -1]
    return data, target


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
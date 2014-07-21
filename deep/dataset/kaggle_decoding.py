"""
Kaggle Decoding Data Class
"""

import numpy
from scipy.io import loadmat
from dataset import DataSet


class KaggleDecoding(DataSet):
    """ Kaggle Decoding dataset from <link> """

    def __init__(self):

        print "loading kaggle decoding dataset ... ",
        data = self.load()
        print "ok"

        self.train_set = self.reshape(data[0])
        self.valid_set = self.reshape(data[1])
        self.test_set = self.reshape(data[2])

        print
        print "x dimensions ...", self.train_set[0][0].shape
        print "y dimensions ...", self.train_set[1][0].shape

    def load(self):

        train_subject_range = range(1, 2)
        valid_subject_range = range(11, 12)
        test_subject_range = range(17, 18)

        train_x = []
        train_y = []
        valid_x = []
        valid_y = []
        test_x = []
        test_y = []

        for subject in train_subject_range:
            filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            train_x.append(data['X'])
            train_y.append(data['y'])

        for subject in valid_subject_range:
            filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            valid_x.append(data['X'])
            valid_y.append(data['y'])

        for subject in test_subject_range:
            filename = 'data/kaggle_decoding/test_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            test_x.append(data['X'])
            test_y.append(data['Id'])

        train_x = numpy.vstack(train_x)
        train_y = numpy.concatenate(train_y)
        valid_x = numpy.vstack(valid_x)
        valid_y = numpy.concatenate(valid_y)
        test_x = numpy.vstack(test_x)
        test_y = numpy.concatenate(test_y)

        return ((train_x, train_y),
                (valid_x, valid_y),
                (test_x, test_y))

k = KaggleDecoding()
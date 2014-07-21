"""
Kaggle Decoding Data Class
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
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
        print

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
            filename = '../../deep/dataset/data/kaggle_decoding/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            train_x.append(data['X'])
            train_y.append(data['y'])

        for subject in valid_subject_range:
            filename = '../../deep/dataset/data/kaggle_decoding/train_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            valid_x.append(data['X'])
            valid_y.append(data['y'])

        for subject in test_subject_range:
            filename = '../../deep/dataset/data/kaggle_decoding/test_subject%02d.mat' % subject
            data = loadmat(filename, squeeze_me=True)

            test_x.append(data['X'])
            test_y.append(data['Id'])

        train_x = np.vstack(train_x)
        train_y = np.concatenate(train_y)
        valid_x = np.vstack(valid_x)
        valid_y = np.concatenate(valid_y)
        test_x = np.vstack(test_x)
        test_y = np.concatenate(test_y)

        return ((train_x, train_y),
                (valid_x, valid_y),
                (test_x, test_y))

    def best_sensor(self, subject_set):
        """ most discriminative sensor based on logistic regression """

        # TODO: refactor

        data = self.load()

        X = data[0][0]
        y = data[0][1]

        clf = LogisticRegression(random_state=0)
        score_channel = np.zeros(306)

        for channel in range(306):
            X_channel = X[:,]
            X_channel = X[:,channel,:].copy()
            X_channel -= X_channel.mean(0)
            X_channel = np.nan_to_num(X_channel / X_channel.std(0))
            scores = cross_val_score(clf, X_channel, y, cv=3, scoring='accuracy')
            score_channel[channel] = scores.mean()
            print score_channel[channel]

        best_channel = np.argsort(score_channel)[-1]

        print best_channel

        return X[:, best_channel, :]

    def plot_classes(self):
        """ plot single and averaged of best sensor """

        # TODO: pickle data with best sensors to avoid searching each time
        # TODO: plot best channel

        best_channel = 277

        train_x = self.train_set[0]
        train_y = np.argmax(self.train_set[1], axis=1).reshape(594)
        train_x = train_x[:, best_channel, :].reshape((594,375))

        plt.subplot(121)
        plt.plot(train_x[train_y == 1].mean(0))
        plt.subplot(122)
        plt.plot(train_x[train_y == 0].mean(0))
        plt.show()



k = KaggleDecoding()
k.plot_classes()


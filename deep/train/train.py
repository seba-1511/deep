"""
Model Parameter Selection
"""

import numpy as np


def sgd(train_set, model, unsupervised=False):
    """ stochastic gradient descent """

    train_x, train_y = train_set

    if unsupervised:
        train_y = train_x

    epochs = 1000
    for epoch in range(epochs):
        total_error = 0.0
        for x, y in zip(train_x, train_y):
            error = model.fprop(x) - y
            total_error += np.sum(error**2)
            model.bprop(error)
            model.update(.001)

        print total_error


def bgd(train_set, model, unsupervised=False):

    # TODO: implement batch gradient descent

    raise NotImplementedError


def gd(train_set, model, unsupervised=False):

    # TODO: implement gradient descent

    raise NotImplementedError


def cgd(train_set, model, unsupervised=False):

    # TODO: implement conjugate gradient descent

    raise NotImplementedError


def score(valid_set, model):

        valid_x, valid_y = valid_set

        correct = 0.0
        for x, y in zip(valid_x, valid_y):
            if np.argmax(model.fprop(x)) == np.argmax(y):
                correct += 1.0
        print correct / len(valid_x)

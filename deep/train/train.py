"""
Model Parameter Selection
"""

import numpy as np


def sgd(dataset, model, unsupervised=False):
    """ stochastic gradient descent """

    bgd(dataset, model, batch_size=1)


def bgd(dataset, model, batch_size=500):
    """ batch gradient descent """

    for epoch in range(1):

        for i in range(0, dataset.train_size, batch_size):

            batch_x = dataset.train_x[i:i+batch_size]
            batch_y = dataset.train_bin_y[i:i+batch_size]

            error = model.fprop(batch_x) - batch_y
            print "batch error", np.sum(error**2)
            model.bprop(error)
            model.update(10)


def score(dataset, model):

        guess = np.argmax(model.fprop(dataset.valid_x), axis=1)
        correct = dataset.valid_y

        total = np.sum(guess == correct)

        print float(total) / dataset.valid_size
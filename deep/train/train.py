"""
Model Parameter Selection
"""

import numpy as np

def sgd(dataset, model, unsupervised=False):
    """ stochastic gradient descent """

    bgd(dataset, model, batch_size=1)


def bgd(dataset, model, batch_size = 500):
    """ batch gradient descent """

    train_x = dataset.data[0][0]
    train_y = dataset.reshape_y(dataset.data[0][1])

    num_train = train_x.shape[0]

    for epoch in range(1):

        for i in range(0, num_train, batch_size):

            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            error = model.fprop(batch_x) - batch_y
            model.bprop(error)
            model.update(.1)


def score(dataset, model):

        valid_x, valid_y = dataset.data[0]
        valid_y = dataset.reshape_y(valid_y)

        guess = np.argmax(model.fprop(valid_x), axis=1)
        correct = np.argmax(valid_y, axis=1)

        total = np.sum(guess == correct)

        print float(total) / len(valid_x)
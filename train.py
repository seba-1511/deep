"""
Model Parameter Selection
"""

import numpy as np
import mlp
import load_data


def sgd(train_set, model, unsupervised=False):
    """ stochastic gradient descent """

    train_x, train_y = train_set

    if unsupervised:
        train_y = train_x

    epochs = 1

    for epoch in range(epochs):

        total_error = 0.0

        for x, y in zip(train_x, train_y):

            error = model.fprop(x) - y

            total_error += np.sum(error**2)

            model.bprop(error)
            model.update()

        print total_error


def score(valid_set, model):

        valid_x, valid_y = valid_set

        correct = 0.0

        for x, y in zip(valid_x, valid_y):

            if np.argmax(model.fprop(x)) == np.argmax(y):

                correct += 1.0

        print correct / len(valid_x)

data = load_data.mnist()
data = load_data.reshape(data)

layer1 = mlp.SigmoidLayer(784, 30)
layer2 = mlp.SigmoidLayer(30, 10)
layers = [layer1, layer2]

mlp = mlp.MLP(layers)

sgd(data[0], mlp)
score(data[1], mlp)

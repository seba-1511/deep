# -*- coding: utf-8 -*-
"""
    deep.networks.base
    ---------------------

    Implements the feed forward neural network model.

    :references: theano deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from deep.layers import Layer
from deep.base import Supervised
from deep.fit.base import Iterative
from deep.activations import Softmax
from deep.costs.base import NegativeLogLikelihood
from deep.updates.base import GradientDescent


class NN(Supervised):
    """A Feed Forward Neural Network is composed of one or more layers,
    each parametrized by a weight matrix and a biases vector. The weights
    are learned, typically through an iterative optimization procedure, and
    then the network can be used to make predictions on new data.

    Example::

        from deep.datasets import load_mnist
        mnist = load_mnist()
        X_train, y_train = mnist[0]
        X_test, y_test = mnist[2]

        from deep.networks import FeedForwardNN
        clf = FeedForwardNN()

        clf.fit(X_train, y_train)
        print clf.costs(X_test, y_test)

    :references:

    :param activations: activations for each layer.
    :param layer_sizes: number of units in hidden layers.
    :param learning_rate: step size used in fit.
    :param n_iter: number of iterations to run fit.
    :param batch_size: size of batches used for fit.
    :param cost: the function that defines the training error.
    :param update: the way that the parameters are updated.
    :param costs: the costs that is printed during training.
    :param fit: the fit method to use when calling fit().
    """
    def __init__(self, layers=None, learning_rate=10, cost=NegativeLogLikelihood(),
                 update=GradientDescent(), fit=Iterative()):

        self.layers = layers or []
        self.learning_rate = learning_rate
        self.fit_method = fit
        self.cost = cost
        self.update = update

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @property
    def updates(self):
        cost = self.cost(self._symbolic_predict_proba(self.x), self.y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_predict_proba(self, X):
        """A Theano expression representing a class distribution."""
        n_samples, n_features = X.shape

        import theano.tensor as T
        from deep.layers import ConvolutionLayer
        if isinstance(self.layers[0], ConvolutionLayer):
            dim = T.cast(T.sqrt(n_features), dtype='int64')
            X = X.reshape((n_samples, 1, dim, dim))

        for layer in self.layers:

            if not isinstance(layer, ConvolutionLayer):
                X = X.flatten(2)

            X = layer._symbolic_transform(X)

        return X

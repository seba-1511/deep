# -*- coding: utf-8 -*-
"""
    deep.models.base
    ---------------------

    Implements the feed forward neural network model.

    :references: theano deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

import theano.tensor as T
from theano import function

from deep.fit.base import Iterative
from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.updates.base import GradientDescent
from deep.datasets import SupervisedData

class NN(object):
    """A Feed Forward Neural Network is composed of one or more layers,
    each parametrized by a weight matrix and a biases vector. The weights
    are learned, typically through an iterative optimization procedure, and
    then the network can be used to make predictions on new data.

    Example::

        from deep.datasets import load_mnist
        mnist = load_mnist()
        X_train, y_train = mnist[0]
        X_test, y_test = mnist[2]

        from deep.models import FeedForwardNN
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

    x = T.matrix()
    y = T.lvector()
    _predict_proba_function = None

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @property
    def updates(self):
        cost = self.cost(self._symbolic_predict_proba(self.x), self.y)

        #: add regularizer to network

        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        #: compile these in fit method
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def _symbolic_predict(self, x):
        return T.argmax(self._symbolic_predict_proba(x), axis=1)

    def _symbolic_predict_proba(self, X):
        """A Theano expression representing a class distribution."""
        for layer in self.layers:
            X = layer._symbolic_transform(X)
        return X

    def _symbolic_score(self, x, y):
        cost = PredictionError()
        return cost(self._symbolic_predict(x), y)

    def fit(self, X, y=None):
        #: should we just remove X, y and take a dataset?
        if not isinstance(X, SupervisedData):
            dataset = SupervisedData(X, y)
        else:
            dataset = X

        X = dataset.batch(1)

        for layer in self.layers:
            X = layer.fit_transform(X)

        return self.fit_method(self, dataset)

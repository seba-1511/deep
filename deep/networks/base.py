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
import theano.tensor as T

from sklearn.base import ClassifierMixin
from deep.base import LayeredModel

from theano import shared, config, function

from deep.fit.base import Iterative
from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.layers.base import Layer
from deep.updates.base import GradientDescent
from deep.datasets.base import Data
from deep.activations.base import Sigmoid, Softmax


class FeedForwardNN(LayeredModel, ClassifierMixin):
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
    def __init__(self, layers=None, learning_rate=10, batch_size=100,
                 _cost=NegativeLogLikelihood(), update=GradientDescent(),
                 fit=Iterative(), _score=PredictionError()):

        self.layers = layers or []
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        #: better names for _params
        #:
        #: changed (fit, cost, score) to (_fit, _cost, _score) otherwise
        #: getattr grabs the method by the same name and leads to recursion
        self.fit_method = fit
        self._cost = _cost
        self._score = _score
        self.update = update

        self.x = T.matrix()
        self.y = T.lvector()

        self.data = None

    _score_function = None
    _predict_function = None
    _predict_proba_function = None

    @property
    def updates(self):
        cost = self._symbolic_cost(self.x, self.y)
        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def predict(self, X):
        if not self._predict_function:
            self._predict_function = function([self.x], self._symbolic_predict(self.x))
        return self._predict_function(X)

    def _symbolic_predict(self, X):
        """A Theano expression representing a class prediction."""
        return T.argmax(self._symbolic_predict_proba(X), axis=1)

    def predict_proba(self, X):
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def _symbolic_predict_proba(self, X):
        """A Theano expression representing a class distribution."""
        for layer in self:
            X = layer._symbolic_transform(X)
        return X

    def _symbolic_cost(self, X, y):
        cost = self._cost(self._symbolic_predict_proba(X), y)
        for layer in self:
            if layer.regularization is not None:
                cost += layer.regularization(layer)
        return cost

    def score(self, X, y):
        X = np.asarray(X, dtype=config.floatX)
        if not self._score_function:
            self._score_function = function([self.x, self.y], self._symbolic_score(self.x, self.y))
        return self._score_function(X, y)

    def _symbolic_score(self, X, y):
        return self._score(self._symbolic_predict(X), y)

    def _fit(self, X, y):

        x = X[:1]
        for layer in self:
            x = layer.fit_transform(x)

        #: want to fit last layer to classes
        #: should we let user to this or keep as is?
        n_classes = len(np.unique(y))

        softmax = Layer(n_classes, Softmax())
        softmax.fit(x)
        self.layers.append(softmax)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=config.floatX)

        #: fit_method call _fit to get around
        #: data resizing during augmentation
        #: (check fit method for output size)
        self.fit_method(self, X, y)

        #: hack to get clean predictions after training
        #: this fails if we retrain the model since it won't
        #: have the original corruption.
        for layer in self:
            layer.corruption = None
        return self

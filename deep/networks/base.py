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

from deep.fit.base import Fit
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
    def __init__(self, layers=(100,), activation=Sigmoid(),
                 learning_rate=10, n_iter=10, batch_size=100,
                 _cost=NegativeLogLikelihood(), update=GradientDescent(),
                 _fit=Fit(), _score=PredictionError(), corruption=None, regularization=None):

        #: fix this

        #: if layers arg is a list of Layer classes, use as self.layers
        are_layer_classes = [isinstance(layer, Layer) for layer in layers]
        if all(are_layer_classes):
            self.layers = layers
        #: otherwise, we init self.layers in the fit method
        else:
            self.layer_sizes = list(layers) #: fix this
            self.layers = []

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        #: better names for _params
        #:
        #: changed (fit, cost, score) to (_fit, _cost, _score) otherwise
        #: getattr grabs the method by the same name and leads to recursion
        self._fit = _fit
        self._cost = _cost
        self._score = _score
        self.update = update
        self.activation = activation
        self.corruption = corruption
        self.regularization = regularization

        self.x = T.matrix()
        self.y = T.lvector()
        self.i = T.lscalar()

        self._fit_function = None
        self._score_function = None
        self._predict_function = None
        self._predict_proba_function = None
        self.data = None

    @property
    def givens(self):
        """A dictionary mapping Theano var to data."""

        #: is there a way to move this to the LayeredModel class?

        X = shared(np.asarray(self.data.X, dtype=config.floatX))
        y = shared(np.asarray(self.data.y, dtype='int64'))
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        return {self.x: X[batch_start:batch_end],
                self.y: y[batch_start:batch_end]}

    @property
    def updates(self):
        """Collects the updates for each param in each layer."""
        rv = list()
        for param in self.params:
            cost = self._symbolic_cost(self.x, self.y)
            if self.regularization is not None:
                cost += self.regularization(param)
            updates = self.update(cost, param, self.learning_rate)
            for update in updates:
                rv.append(update)
        return rv


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
        return self._cost(self._symbolic_predict_proba(X), y)

    def score(self, X, y):
        X = np.asarray(X, dtype=config.floatX)
        if not self._score_function:
            self._score_function = function([self.x, self.y], self._symbolic_score(self.x, self.y))
        return self._score_function(X, y)

    def _symbolic_score(self, X, y):
        return self._score(self._symbolic_predict(X), y)

    def fit(self, X, y):

        #: this sucks (figure out how to remove data)

        if not self.data:
            self.data = Data(X, y)
        elif self.data != Data(X, y):
            self.data = Data(X, y)
            self._fit_function = None
        if not self.layers:
            #: merge this with conv init

            #: better name for dummy batch
            dummy_batch = np.zeros((self.batch_size, self.data.features), dtype=config.floatX)

            #: init layers
            for layer_size in self.layer_sizes:
                size = (dummy_batch.shape[1], layer_size)
                layer = Layer(size, self.activation, self.corruption)
                self.layers.append(layer)
                dummy_batch = layer.transform(dummy_batch)

            #: init softmax layer
            size = (dummy_batch.shape[1], self.data.classes)
            self.layers.append(Layer(size, Softmax()))

        self._fit(self)

        #: hack to get clean predictions after training
        #: this fails if we retrain the model since it won't
        #: have the original corruption.
        for layer in self:
            layer.corruption = None
        return self

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

class NN(object):

    def __init__(self, layers, learning_rate=10, update=GradientDescent(),
                 fit=Iterative(), cost=NegativeLogLikelihood(), regularize=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.update = update
        self.fit_method = fit
        self.cost = cost
        self.regularize = regularize

    #: this is only used to compile predict_proba
    #: do this in fit?
    x = T.matrix()
    _predict_proba_function = None

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    #: can we move this to updates?
    def updates(self, x, y):
        cost = self.cost(self._symbolic_predict_proba(x), y)

        for param in self.params:
            cost += self.regularize(param)

        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_predict(self, x):
        return T.argmax(self._symbolic_predict_proba(x), axis=1)

    def _symbolic_predict_proba(self, X):
        for layer in self.layers:
            X = layer._symbolic_transform(X)
        return X

    def _symbolic_score(self, x, y):
        cost = PredictionError()
        return cost(self._symbolic_predict(x), y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        #: compile these in fit method
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X, y=None):
        x = X[:1]
        for layer in self.layers:
            x = layer.fit_transform(x)
        return self.fit_method.fit(self, X, y)

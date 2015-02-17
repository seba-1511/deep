# -*- coding: utf-8 -*-
"""
    deep.autoencoders.base
    ---------------------

    Implements a tied autoencoders.

    :references: deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

from theano import function
from deep.fit import Iterative
from deep.costs import BinaryCrossEntropy
from deep.updates import GradientDescent


class AE(object):

    def __init__(self, encoder, decoder, learning_rate=10, update=GradientDescent(),
                 fit=Iterative(), cost=BinaryCrossEntropy()):
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.update = update
        self.fit_method = fit
        self.cost = cost

        #: move this to fit
        from theano.tensor import matrix
        self.x = matrix()

    _transform_function = None
    _inverse_transform_function = None

    @property
    def params(self):
        return [param for layer in self.encoder + self.decoder for param in layer.params]

    def updates(self, x):
        cost = self._symbolic_score(x)

        #: add regularizer to network

        updates = list()
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    def _symbolic_transform(self, x):
        for layer in self.encoder:
            x = layer._symbolic_transform(x)
        return x

    def _symbolic_inverse_transform(self, x):
        for layer in self.decoder:
            x = layer._symbolic_transform(x)
        return x

    def _symbolic_score(self, x):
        reconstruct = self._symbolic_inverse_transform(self._symbolic_transform(x))
        return self.cost(reconstruct, x)

    #: compile this in fit
    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    #: compile this in fit
    def inverse_transform(self, X):
        #: compile these in fit method
        if not self._inverse_transform_function:
            self._inverse_transform_function = function([self.x], self._symbolic_inverse_transform(self.x))
        return self._inverse_transform_function(X)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def score(self, X, y):
        #: add transform/_symbolic api to costs and use cross_entropy here
        raise NotImplementedError

    def fit(self, X, y=None):
        x = X[:1]
        for layer in self.encoder + self.decoder:
            x = layer.fit_transform(x)
        return self.fit_method.fit(self, X, y)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
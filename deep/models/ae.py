# -*- coding: utf-8 -*-
"""
    deep.autoencoders.base
    ---------------------

    Implements a tied autoencoders.

    :references: deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from theano import function

from deep.fit import Iterative
from deep.costs import BinaryCrossEntropy
from deep.updates import GradientDescent
from deep.datasets import UnsupervisedData

class AE(object):

    def __init__(self, encoder, decoder, learning_rate=10, update=GradientDescent(),
                 fit=Iterative(), cost=BinaryCrossEntropy()):
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.update = update
        self.fit_method = fit
        self.cost = cost

    _transform_function = None
    _inverse_transform_function = None

    @property
    def params(self):
        return [param for layer in self.encoder + self.decoder for param in layer.params]

    def updates(self, x):
        reconstruct = self._symbolic_inverse_transform(self._symbolic_transform(x))
        cost = self.cost(reconstruct, x)

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

    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

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
        if not isinstance(X, UnsupervisedData):
            dataset = UnsupervisedData(X)
        else:
            dataset = X

        X = dataset.batch(1)

        for layer in self.encoder + self.decoder:
            X = layer.fit_transform(X)

        return self.fit_method(self, dataset)

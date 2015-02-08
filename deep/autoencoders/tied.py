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
import theano.tensor as T

from sklearn.base import TransformerMixin

from theano import function, config, shared

from deep.layers.base import InvertibleLayer, Layer
from deep.base import LayeredModel
from deep.activations.base import Sigmoid
from deep.fit.base import Fit
from deep.datasets.base import Data
from deep.costs.base import BinaryCrossEntropy
from deep.updates.base import GradientDescent


class TiedAE(LayeredModel, TransformerMixin):
    """ """
    def __init__(self, layers=(100,), activation=Sigmoid(), learning_rate=10,
                 n_iter=10, batch_size=100, _fit=Fit(),
                 cost=BinaryCrossEntropy(), update=GradientDescent(),
                 corruption=None):

        self.layers = []
        for layer in layers:
            if isinstance(layer, int):
                layer = InvertibleLayer(layer, activation, corruption)
            self.layers.append(layer)


        #: hyperparams (vars)
        self.activation = activation
        self.corruption = corruption
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size

        #: hyperparams (funcs)
        self._fit = _fit
        self.cost = cost
        self.update = update

        #: symbolic vars
        self.x = T.matrix()
        self.i = T.iscalar()

        #: place holders
        self.data = None
        self._fit_function = None
        self._score_function = None
        self._transform_function = None
        self._inverse_transform_function = None

    @property
    def updates(self):
        """"""
        updates = list()
        cost = self._symbolic_score(self.x)
        for param in self.params:
            updates.extend(self.update(cost, param, self.learning_rate))
        return updates

    @property
    def givens(self):
        """"""
        #: is there a way to move this to a multilayermodel class?
        X = shared(np.asarray(self.data.X, dtype=config.floatX))
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        return {self.x: X[batch_start:batch_end]}

    @property
    def fit_function(self):
        """"""
        if not self._fit_function:
            self._fit_function = function(inputs=[self.i],
                                          outputs=self._symbolic_score(self.x),
                                          updates=self.updates,
                                          givens=self.givens)
        return self._fit_function

    def score(self, X):
        if not self._score_function:
            self._score_function = function([self.x], self._symbolic_score(self.x))
        return self._score_function(X)

    def _symbolic_score(self, X):
        return self.cost(self._symbolic_reconstruct(X), X)

    def transform(self, X):
        for layer in self:
            X = layer.transform(X)
        return X

    def _symbolic_transform(self, X):
        for layer in self:
            X = layer._symbolic_transform(X)
        return X

    def inverse_transform(self, X):
        for layer in self[::-1]:
            X = layer.inverse_transform(X)
        return X

    def _symbolic_inverse_transform(self, X):
        for layer in reversed(self):
            X = layer._symbolic_inverse_transform(X)
        return X

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def _symbolic_reconstruct(self, X):
        return self._symbolic_inverse_transform(self._symbolic_transform(X))

    def fit(self, X, y=None):
        if not self.data:
            self.data = Data(X)

        for layer in self:
            X = layer.fit_transform(X)

        return self._fit(self)

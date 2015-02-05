# -*- coding: utf-8 -*-
"""
    deep.autoencoder.multilayer
    ---------------------------

    Implements a multilayer autoencoder.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T
from theano import shared, function, config

from sklearn.base import TransformerMixin
from deep.base import LayeredModel

from deep.autoencoder.base import TiedAE
from deep.fit.base import Fit
from deep.costs.base import BinaryCrossEntropy
from deep.utils.base import theano_compatible
from deep.updates.base import GradientDescent
from deep.datasets.base import Data
from deep.activations.base import Sigmoid


class MultilayerAE(LayeredModel, TransformerMixin):

    def __init__(self, layers=(100, 100), activation=Sigmoid(),
                 learning_rate=10, n_iter=10, batch_size=100,
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 _fit=Fit()):

        self.layer_sizes = list(layers)
        self.layers = []

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._fit = _fit
        self._cost = _cost
        self.update = update
        self.activation = activation

        self.x = T.matrix()
        self.y = self.x
        self.i = T.lscalar()

        self._fit_function = None
        self.data = None

    @property
    def givens(self):
        """A dictionary mapping Theano var to data."""
        X = shared(np.asarray(self.data.X, dtype=config.floatX))
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        return {self.x: X[batch_start:batch_end]}

    @theano_compatible
    def transform(self, X):
        for autoencoder in self:
            X = autoencoder.transform(X)
        return X

    @theano_compatible
    def inverse_transform(self, X):
        for autoencoder in self[::-1]:
            X = autoencoder.inverse_transform(X)
        return X

    @theano_compatible
    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    @theano_compatible
    def cost(self, X, y):
        return self._cost(self.reconstruct(X), y)

    @property
    def fit_function(self):
        """The compiled Theano function used to train the network."""
        if not self._fit_function:
            self._fit_function = function(inputs=[self.i],
                                          outputs=self.cost(self.x, self.y),
                                          updates=self.updates,
                                          givens=self.givens)
        return self._fit_function

    @theano_compatible
    def score(self, X, y):
        return self._cost(self.reconstruct(X), X)

    def fit(self, X):
        if not self.data:
            self.data = Data(X)
        for size in self.layer_sizes:
            #: fit with zero iters just to init layer shapes (sketch)
            self.layers.append(TiedAE(self.activation, self.learning_rate, size,
                                      0, self.batch_size, self._fit, self._cost,
                                      self.update))
        #: transform X through ae's to set each layer size (even sketchier)
        for autoencoder in self:
            X = autoencoder.fit(X).transform(X)

        return self._fit(self)

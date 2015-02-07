# -*- coding: utf-8 -*-
"""
    deep.autoencoders.stacked
    ------------------------

    Implements a stacked autoencoders.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

from sklearn.base import TransformerMixin
from deep.base import LayeredModel

from deep.autoencoders.base import TiedAE
from deep.fit.base import Fit
from deep.costs.base import BinaryCrossEntropy
from deep.utils.base import theano_compatible
from deep.updates.base import GradientDescent
from deep.activations.base import Sigmoid


class StackedAE(LayeredModel, TransformerMixin):

    def __init__(self, layers=(100, 100), activation=Sigmoid(),
                 learning_rate=1, n_iter=10, batch_size=100,
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 _fit=Fit()):

        self.layer_sizes = list(layers)
        self.layers = []

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._fit = _fit
        self._cost = _cost
        self.update= update
        self.activation = activation

        self._fit_function = None
        self.data = None

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @theano_compatible
    def transform(self, X):
        for autoencoder in self:
            X = autoencoder.transform(X)
        return X

    @theano_compatible
    def inverse_transform(self, X):
        for autoencoder in self[::-1]:
            X = autoencoder.inverse_transfom
        return X

    def fit(self, X):
        for size in self.layer_sizes:
            self.layers.append(TiedAE(self.activation, self.learning_rate, size,
                                      self.n_iter, self.batch_size, self._fit, self._cost,
                                      self.update))
        for autoencoder in self:
            X = autoencoder.fit(X).transform(X)
        return self

# -*- coding: utf-8 -*-
"""
    deep.autoencoders.denoising
    --------------------------

    Implements a denoising autoencoders.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import theano.tensor as T

from deep.autoencoders.multilayer import MultilayerAE
from deep.autoencoders.stacked import StackedAE
from deep.autoencoders.base import TiedAE
from deep.layers import DenoisingLayer
from deep.corruptions.base import SaltAndPepper
from deep.utils.base import theano_compatible
from deep.activations.base import Sigmoid
from deep.fit.base import Fit
from deep.costs.base import BinaryCrossEntropy
from deep.updates.base import GradientDescent
from deep.datasets.base import Data


class TiedDenoisingAE(TiedAE, DenoisingLayer):

    def __init__(self, activation=Sigmoid(), learning_rate=10,
                 n_hidden=100, n_iter=10, batch_size=100, _fit=Fit(),
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 corruption=SaltAndPepper()):
        super(TiedDenoisingAE, self).__init__(activation, learning_rate,
                                              n_hidden, n_iter, batch_size,
                                              _fit, _cost, update)
        self.corrupt = corruption

    #: can we push this down into the denoising layer? (mro is currently wrong)
    @theano_compatible
    def score(self, X):
        return self._cost(self.denoise(X), X)

    @theano_compatible
    def denoise(self, X):
        return self.reconstruct(self.corrupt(X))

    @theano_compatible
    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))


class StackedDenoisingAE(StackedAE):

    def __init__(self, layers=(100, 100), activation=Sigmoid(),
             learning_rate=1, n_iter=10, batch_size=100,
             _cost=BinaryCrossEntropy(), update=GradientDescent(),
             _fit=Fit(), corruption=SaltAndPepper()):

        super(StackedDenoisingAE, self).__init__(layers, activation,
                                                 learning_rate, n_iter, batch_size,
                                                 _cost, update, _fit)
        self.corrupt = corruption

    @theano_compatible
    def denoise(self, X):
        return self.reconstruct(self.corrupt(X))

    def fit(self, X):
        for size in self.layer_sizes:
            self.layers.append(TiedDenoisingAE(self.activation, self.learning_rate, size,
                                      self.n_iter, self.batch_size, self._fit, self._cost,
                                      self.update))
        for autoencoder in self:
            X = autoencoder.fit(X).transform(X)
        return self


class MultilayerDenoisingAE(MultilayerAE):

    def __init__(self, layers=(100, 100), activation=Sigmoid(),
                 learning_rate=10, n_iter=10, batch_size=100,
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 _fit=Fit(), corruption=SaltAndPepper()):

        super(MultilayerDenoisingAE, self).__init__(layers, activation,
                                                 learning_rate, n_iter, batch_size,
                                                 _cost, update, _fit)
        self.corrupt = corruption

    @theano_compatible
    def denoise(self, X):
        for autoencoder in self:
            X = autoencoder.transform(autoencoder.corrupt(X))
        return self.inverse_transform(X)

    @theano_compatible
    def cost(self, X, y):
        return self._cost(self.denoise(X), y)

    def fit(self, X):
        if not self.data:
            self.data = Data(X)
        for size in self.layer_sizes:
            #: fit with zero iters just to init layer shapes (sketch)
            self.layers.append(TiedDenoisingAE(self.activation, self.learning_rate, size,
                                      0, self.batch_size, self._fit, self._cost,
                                      self.update, self.corrupt))
        #: transform X through ae's to set each layer size (even sketchier)
        for autoencoder in self:
            X = autoencoder.fit(X).transform(X)

        return self._fit(self)

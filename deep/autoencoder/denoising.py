# -*- coding: utf-8 -*-
"""
    deep.autoencoder.denoising
    --------------------------

    Implements a denoising autoencoder.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import theano.tensor as T

from deep.autoencoder.base import TiedAE
from deep.layers import DenoisingLayer
from deep.corruptions.base import SaltAndPepper
from deep.utils.base import theano_compatible
from deep.activations.base import Sigmoid
from deep.fit.base import Fit
from deep.costs.base import BinaryCrossEntropy
from deep.updates.base import GradientDescent


class TiedDenoisingAE(TiedAE, DenoisingLayer):

    def __init__(self, activation=Sigmoid(), learning_rate=10,
                 n_hidden=100, n_iter=10, batch_size=100, _fit=Fit(),
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 corruption=SaltAndPepper()):
        super(TiedDenoisingAE, self).__init__(activation, learning_rate,
                                              n_hidden, n_iter, batch_size,
                                              _fit, _cost, update)
        self.corrupt = corruption

    #: can we push this down into the layer? (mro is wrong)
    @theano_compatible
    def transform(self, X, noisy=True):
        if noisy:
            X = self.corrupt(X)
        return self.activation(T.dot(X, self.W) + self.b)

    @theano_compatible
    def reconstruct(self, X, noisy=True):
        return self.inverse_transform(self.transform(X))

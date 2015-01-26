"""
Multilayer Autoencoder
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import theano.tensor as T

from theano import function
from theano import shared

from deep.layers.base import SigmoidLayer
from deep.fit.base import simple_batch_gradient_descent
from deep.hyperparams import layer_sizes, batch_size, learning_rate


class MultilayerAE(object):
    """ deep autoencoder_old without greedy layer training? """
    def __init__(self, layer_sizes=layer_sizes, batch_size=batch_size,
                 learning_rate=learning_rate):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def _fit(self, X):
        i = T.lscalar()
        x = T.fmatrix()

        encode = lambda input, layer: layer.__call__(input)
        decode = lambda input, layer: layer.inverse_transform(input)
        transform = reduce(encode, self.layers, x)
        inverse_transform = reduce(decode, self.layers[::-1], x)
        reconstruct = reduce(decode, self.layers[::-1], transform)

        cost = T.mean(T.nnet.binary_crossentropy(reconstruct, x))
        updates = [(param, param - self.learning_rate * T.grad(cost, param))
                   for param in self.params]

        X = shared(np.asarray(X))
        givens = {x: X[i * batch_size: (i + 1) * batch_size]}

        self.transform = function([x], transform)
        self.inverse_transform = function([x], inverse_transform)
        self._fit = function([i], cost, None, updates, givens)

    def fit(self, X):
        n_samples, n_features = X.shape
        n_batches = n_samples / self.batch_size

        for shape in zip([n_features] + self.layer_sizes, self.layer_sizes):
            self.layers.append(SigmoidLayer(shape))

        self._fit(X)

        return simple_batch_gradient_descent(self, n_batches)

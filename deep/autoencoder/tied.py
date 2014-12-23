"""
Tied Autoencoder
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import theano.tensor as T

from theano import function
from theano import shared

from deep.layers.base import Layer
from deep.activations.base import Sigmoid
from deep.fit.base import simple_batch_gradient_descent
from deep.hyperparams import batch_size, learning_rate, n_hidden, n_iter


class TiedAE(Layer):
    """ """
    def __init__(self, activation=Sigmoid, learning_rate=learning_rate,
                 n_hidden=n_hidden, n_iter=n_iter, batch_size=batch_size):
        self.n_hidden = n_hidden
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size

    @property
    def params(self):
        return self.W, self.b, self.b_decode

    def _fit(self, X):
        i = T.lscalar()
        x = T.dmatrix()

        inverse_transform = lambda input: self.activation(T.dot(input, self.W.T) + self.b_decode)
        reconstruct = inverse_transform(self.transform(x))
        cost = T.mean(T.nnet.binary_crossentropy(reconstruct, x))
        updates = [(param, param - self.learning_rate * T.grad(cost, param))
                   for param in self.params]

        X = shared(np.asarray(X, dtype='float64'))
        givens = {x: X[i * self.batch_size: (i + 1) * self.batch_size]}

        self.transform = function([x], self.transform(x))
        self.inverse_transform = function([x], inverse_transform(x))
        self.train = function([i], cost, None, updates, givens)

    def fit(self, X):
        """"""
        n_samples, n_features = X.shape
        n_batches = n_samples / self.batch_size

        super(TiedAE, self).__init__((n_features, n_hidden), self.activation)
        self.b_decode = shared(np.zeros(n_features))

        self._fit(X)

        return simple_batch_gradient_descent(self, n_batches)

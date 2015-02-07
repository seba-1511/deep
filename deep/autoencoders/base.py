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

from sklearn.base import BaseEstimator, ClassifierMixin

from theano import function, config, shared

from deep.layers.base import Layer
from deep.activations.base import Sigmoid
from deep.fit.base import Fit
from deep.datasets.base import Data
from deep.costs.base import BinaryCrossEntropy
from deep.updates.base import GradientDescent


class TiedAE(Layer, BaseEstimator, ClassifierMixin):
    """ """
    def __init__(self, activation=Sigmoid(), learning_rate=10,
                 n_hidden=100, n_iter=10, batch_size=100, _fit=Fit(),
                 _cost=BinaryCrossEntropy(), update=GradientDescent(),
                 corruption=None):

        #: hyperparams (vars)
        self.n_hidden = n_hidden
        self.activation = activation
        self.corruption = corruption
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size

        #: hyperparams (funcs)
        self._fit = _fit
        self._cost = _cost
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
    def params(self):
        return self.W, self.b, self.b_decode

    @property
    def updates(self):
        """"""
        rv = list()
        for param in self.params:
            cost = self._symbolic_score(self.x)
            updates = self.update(cost, param, self.learning_rate)
            for update in updates:
                rv.append(update)
        return rv

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
        return self._cost(self._symbolic_reconstruct(X), X)

    def inverse_transform(self, X):
        if not self._inverse_transform_function:
            self._inverse_transform_function = function([self.x], self._symbolic_inverse_transform(self.x))
        return self._inverse_transform_function(X)

    def _symbolic_inverse_transform(self, X):
        return self.activation(T.dot(X, self.W.T) + self.b_decode)

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def _symbolic_reconstruct(self, X):
        if self.corruption is not None:
            X = self.corruption(X)
        return self._symbolic_inverse_transform(self._symbolic_transform(X))

    def fit(self, X, y=None):
        self.data = Data(X)

        size = self.data.features, self.n_hidden
        super(TiedAE, self).__init__(size, self.activation, self.corruption)
        self.b_decode = shared(np.zeros(self.data.features, dtype=config.floatX))

        self._fit(self)

        #: hack to get clean reconstruction after training.
        self.corruption = None

        return self
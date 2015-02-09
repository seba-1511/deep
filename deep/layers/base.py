# -*- coding: utf-8 -*-
"""
    deep.layers.base
    ----------------

    Implements different layer classes.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from theano import shared, config, function
from theano.tensor.nnet import conv2d

from deep.activations.base import Sigmoid


class Layer(object):
    """An abstract class that represents a neural network layer. Is used
    implicitly by models and can also be used explicity to create
    custom architectures.

    :param shape: a tuple (input_size, output_size).
    :param activation: the activation function to apply after linear transform.
    """
    def __init__(self, n_hidden=100, activation=Sigmoid(), corruption=None, regularization=None):
        self.b = shared(np.zeros(n_hidden, dtype=config.floatX))
        self._transform_function = None
        self.activation = activation
        self.corruption = corruption
        self.regularization = regularization
        self.n_hidden = n_hidden
        self.x = T.matrix()

    @property
    def params(self):
        return self.W, self.b

    @property
    def shape(self):
        return self.W.get_value().shape

    def transform(self, X):
        """ """
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, X):
        if self.corruption is not None:
            X = self.corruption(X)
        return self.activation(T.dot(X, self.W) + self.b)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        #: where should the seed go?
        np.random.seed(1)
        size = X.shape[1], self.n_hidden

        #: change init to 0-1 uniform?
        val = np.sqrt(24. / sum(size))
        self.W = np.random.uniform(low=-val, high=val, size=size)
        self.W = shared(np.asarray(self.W, dtype=config.floatX))
        return self

    def __repr__(self):
        layer_name = str(self.activation) + self.__class__.__name__
        layer_shape = str(self.shape)
        return layer_name + '(shape=' + layer_shape + ')'


#: I hate this. Kill it with fire.
class InvertibleLayer(Layer):

    def __init__(self, n_hidden=100, activation=Sigmoid(), corruption=None, regularization=None):
        self._inverse_transform_function = None
        super(InvertibleLayer, self).__init__(n_hidden, activation, corruption, regularization)

    @property
    def params(self):
        return self.W, self.b, self.b_inverse

    def inverse_transform(self, X):
        """ """
        if not self._inverse_transform_function:
            self._inverse_transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._inverse_transform_function(X)

    def _symbolic_inverse_transform(self, X):
        return self.activation(T.dot(X, self.W.T) + self.b_inverse)

    def fit(self, X):
        n_features = X.shape[1]
        self.b_inverse = shared(np.zeros(n_features, dtype=config.floatX))
        return super(InvertibleLayer, self).fit(X)


class ConvolutionLayer(Layer):
    """An abstract class that represents a convolutional layer. This is called
    implicitely by the ConvolutionalNN class and can also be used explicitly
    to create custom architectures.

    :param filter_size: shape of the convolutional filter
                        (n_filters, n_kernals, height, width).
    :param pool_size: the size of the subsampling pool.
    :param activation: the non-linearly to apply after pooling.
    """
    def __init__(self, n_filters=10, filter_size=5, pool_size=2, activation=Sigmoid(), corruption=None, regularization=None):
        self.b = shared(np.zeros(n_filters, dtype=config.floatX))
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = (pool_size, pool_size)
        self.corruption = corruption
        self.x = T.matrix()
        self.activation = activation
        self.regularization = regularization
        self._transform_function = None

    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, x):
        if x.ndim == 2:
            size = x.shape[1]
            dim = T.cast(T.sqrt(size), dtype='int64')
            x = x.reshape((-1, 1, dim, dim))
        if self.corruption is not None:
            x = self.corruption(x)
        x = conv2d(x, self.W, subsample=self.pool_size)
        return self.activation(x + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)

    def fit(self, X):

        #: where should the seed go?
        np.random.seed(1)
        size = self.n_filters, 1, self.filter_size, self.filter_size

        #: change init to 0-1 uniform?
        val = np.sqrt(24. / sum(size))
        self.W = np.random.uniform(low=-val, high=val, size=size)
        self.W = shared(np.asarray(self.W, dtype=config.floatX))
        return self

    def __repr__(self):
        return self.__class__.__name__
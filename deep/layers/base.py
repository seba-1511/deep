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
from deep.corruptions.base import SaltAndPepper


class Layer(object):
    """An abstract class that represents a neural network layer. Is used
    implicitly by models and can also be used explicity to create
    custom architectures.

    :param shape: a tuple (input_size, output_size).
    :param activation: the activation function to apply after linear transform.
    """
    def __init__(self, size=(784, 100), activation=Sigmoid(), corruption=None):
        np.random.seed(1)
        val = np.sqrt(24. / sum(size))
        self.activation = activation
        self.corruption = corruption
        self.b = shared(np.zeros(size[1], dtype=config.floatX))
        self.W = shared(np.asarray(np.random.uniform(low=-val, high=val, size=size), dtype=config.floatX))
        self.x = T.matrix()
        self._transform_function = None

    @property
    def params(self):
        """The weight and bias of the convolutional layer.

        :rtype : tuple of Theano symbolic variables
        """
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

    def __repr__(self):
        layer_name = str(self.activation) + self.__class__.__name__
        layer_shape = str(self.shape)
        return layer_name + '(shape=' + layer_shape + ')'


class DenoisingLayer(Layer):
    """An abstract class that represents a neural network layer. Is used
    implicitly by models and can also be used explicity to create
    custom architectures.

    :param shape: a tuple (input_size, output_size).
    :param activation: the activation function to apply after linear transform.
    """
    def __init__(self, size=(784, 100), corruption=SaltAndPepper()):
        super(DenoisingLayer, self).__init__(size)
        self.corrupt = corruption

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return super(DenoisingLayer, self).__call__(self.corrupt(x))

    def __repr__(self):
        layer_name = str(self.activation) + self.__class__.__name__
        corruption = str(self.corrupt)
        layer_shape = str(self.shape)
        return layer_name + '(corruption=' + corruption + \
                            ',  shape=' + layer_shape + ')'


class ConvolutionLayer(Layer):
    """An abstract class that represents a convolutional layer. This is called
    implicitely by the ConvolutionalNN class and can also be used explicitly
    to create custom architectures.

    :param filter_size: shape of the convolutional filter
                        (n_filters, n_kernals, height, width).
    :param pool_size: the size of the subsampling pool.
    :param activation: the non-linearly to apply after pooling.
    """
    def __init__(self, filter_size=(10, 1, 5, 5), pool_size=2, activation=Sigmoid()):
        super(ConvolutionLayer, self).__init__(filter_size, activation)
        self.b = shared(np.zeros(filter_size[0], dtype=config.floatX))
        self.pool_size = (pool_size, pool_size)
        self.x = T.tensor4()

    def transform(self, X):
        if not self._transform_function:
            self._transform_function = function([self.x], self._symbolic_transform(self.x))
        return self._transform_function(X)

    def _symbolic_transform(self, x):
        x = conv2d(x, self.W, subsample=self.pool_size)
        return self.activation(x + self.b.dimshuffle('x', 0, 'x', 'x'))

    def __repr__(self):
        return self.__class__.__name__
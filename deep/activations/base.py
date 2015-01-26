# -*- coding: utf-8 -*-
"""
    deep.activations.base
    ---------------------

    Implements various types of activation functions.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import theano.tensor as T
from abc import abstractmethod
from deep.utils.base import theano_compatible


#: this need documentation that explains the @theano_compatible decorator.


class Activation(object):
    """An abstract class that represents an activation function. Once
    initialized, an activation class transforms inputs through its
    __call__ method.

    Example::

        activation = Sigmoid()
        output = activation(input)

    :param corruption_level: the amount of corruption to add to the input.
    :param rng: a Theano RandomStreams() random number generator.
    """
    @abstractmethod
    def __call__(self, X):
        """

        :param X:
        """

    def __repr__(self):
        return str(self.__class__.__name__) + 'Activation'


class Linear(Activation):
    """A linear activation does not transform inputs. It simply passes through
    the values to mirror the behavior of other activations.

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a transformed Theano symbolic of same dims as the input.
    """
    @theano_compatible
    def __call__(self, X):
        return X


class RectifiedLinear(Activation):
    """A rectified linear activation transforms the input by taking the maximum
    of the input or 0 for each value.

    :reference: ? title ? Glorot and Bengio 2011

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a transformed Theano symbolic of same dims as the input.
    """
    @theano_compatible
    def __call__(self, X):
        return T.switch(X > 0.0, X, 0.0)


class Sigmoid(Activation):
    """A sigmoid activation transforms the input by applying the sigmoid
    function element-wise to the values.

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a transformed Theano symbolic of same dims as the input.
    """
    @theano_compatible
    def __call__(self, X):
        return T.nnet.sigmoid(X)


class Softmax(Activation):
    """A softmax activation transforms the input by applying the softmax
    function to the inputs, resulting in outputs that sum to 1.

    :reference:

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a transformed Theano symbolic of same dims as the input.
    """
    @theano_compatible
    def __call__(self, X):
        return T.nnet.softmax(X)


class Tanh(Activation):
    """A tanh activation transforms the input by applying the hyperbolic
    tangent function element-wise to the values.

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a transformed Theano symbolic of same dims as the input.
    """
    @theano_compatible
    def __call__(self, X):
        return T.tanh(X)

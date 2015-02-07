# -*- coding: utf-8 -*-
"""
    deep.costs.base
    ---------------------

    Implements various types of cost functions.

    :references: pylearn2 (cost module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import theano.tensor as T
from abc import abstractmethod

#: should we separate supervised and unsupervised objectives?
#:
#: for example, negative log-likelihood only works for supervised
#: but binary cross-entropy only works for unsupervised


class Cost(object):
    """An abstract class that represents a cost function. Once initialized, a
    cost class returns a Theano symbolic cost through its __call__ method.

    Example::

        cost_function = SquaredError()
        cost = cost_function(input, target)
    """
    @abstractmethod
    def __call__(self, x, y):
        """"""

    def __repr__(self):
        return str(self.__class__.__name__) + 'Cost'


class SquaredError(Cost):
    """Squared error measures the euclidian distance between input and target,
    returning the mean over all examples.

    :reference:

    :param x: a tensor_like Theano symbolic representing the input.
    :param y: a tensor_like Theano symbolic representing the target.
    :return: a Theano expression representing the cost function.
    """

    #: this assumes 1-hot encoding
    #:
    #: how do handle single value y's? Same as NLL?

    def __call__(self, x, y):
        return T.mean(T.sum((x - y) ** 2, axis=-1))


class BinaryCrossEntropy(Cost):
    """Binary cross entropy measures the ???.

    :reference:

    :param x: a tensor_like Theano symbolic representing the input.
    :param y: a tensor_like Theano symbolic representing the target.
    :return: a Theano expression representing the cost function.
    """
    def __call__(self, x, y):
        return T.mean(T.nnet.binary_crossentropy(x, y))


class NegativeLogLikelihood(Cost):
    """Negative log likelihood measures the ???.

    :reference:

    :param x: a tensor_like Theano symbolic representing the input.
    :param y: a tensor_like Theano symbolic representing the target.
    :return: a Theano expression representing the cost function.
    """
    def __call__(self, x, y):
        return -T.mean(T.log(x)[T.arange(y.shape[0]), y])


class PredictionError(Cost):
    """Prediction error measures the number of correct predictions, returning
    the mean over all examples.

    :reference:

    :param x: a tensor_like Theano symbolic representing the input.
    :param y: a tensor_like Theano symbolic representing the target.
    :return: a Theano expression representing the cost function.
    """
    def __call__(self, x, y):
        return T.mean(T.eq(x, y))
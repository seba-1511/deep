# -*- coding: utf-8 -*-
"""
    deep.regularizations.base
    -------------------------

    Implements various types of regularization.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

from abc import abstractmethod

import theano.tensor as T

class Regularization(object):

    def __init__(self, alpha=.00001):
        self.alpha = alpha

    @abstractmethod
    def __call__(self, X):
        """

        :param X:
        """
    def __repr__(self):
        return str(self.__class__.__name__) + 'Regularization'


class L1(Regularization):

    def __call__(self, param):
        return self.alpha * T.sum(abs(param))


class L2(Regularization):

    def __call__(self, param):
        return self.alpha * T.sum(param ** 2)

# -*- coding: utf-8 -*-
"""
    deep.activations.test.test_base
    -------------------------------

    Tests various types of activation functions.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import unittest
import theano.tensor as T
import numpy as np

from sklearn.utils.testing import assert_equal, assert_true
from deep.activations import Linear, RectifiedLinear, Sigmoid, Softmax, Tanh

activations = [Linear, RectifiedLinear, Sigmoid, Softmax, Tanh]


class TestActivations(unittest.TestCase):

    def test_activation_call_method(self):

        for activation in activations:
            activation = activation()

            X = T.dmatrix()
            X_1d = np.empty(10)
            X_2d = np.empty((1, 10))

            assert_true(isinstance(activation(X), T.TensorVariable))

            if isinstance(activation, Softmax):
                #: T.nnet.softmax implementation reshapes (n,) to (1, n)
                assert_equal(activation(X_1d).shape, X_2d.shape)
            else:
                assert_equal(activation(X_1d).shape, X_1d.shape)
            assert_equal(activation(X_2d).shape, X_2d.shape)
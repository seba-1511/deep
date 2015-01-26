# -*- coding: utf-8 -*-
"""
    deep.corruptions.tests.test_base
    --------------------------------

    Tests various types of corruption.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import unittest
import theano.tensor as T
import numpy as np

from sklearn.utils.testing import assert_equal, assert_true
from deep.corruptions import (Binomial, Dropout, Gaussian, SaltAndPepper,
                              Rotate, Shift)


theano_corruptions = [Binomial, Dropout, Gaussian, SaltAndPepper]
non_theano_corruptions = [Rotate, Shift]


class TestCorruptions(unittest.TestCase):

    def test_theano_corruption_call_method(self):

        for corruption in theano_corruptions:
            corruption = corruption()

            X = T.dmatrix()
            X_1d = np.empty(10)
            X_2d = np.empty((1, 10))

            assert_true(isinstance(corruption(X), T.TensorVariable))
            assert_equal(corruption(X_1d).shape, X_1d.shape)
            assert_equal(corruption(X_2d).shape, X_2d.shape)

    def test_non_theano_corruption_call_method(self):

        for corruption in non_theano_corruptions:
            corruption = corruption()
            X_2d = np.empty((1, 10))
            assert_equal(corruption(X_2d).shape, X_2d.shape)


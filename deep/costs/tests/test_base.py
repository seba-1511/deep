# -*- coding: utf-8 -*-
"""
    deep.costs.tests.test_base
    --------------------------

    Tests various types of corruption.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import unittest
import theano.tensor as T
import numpy as np

from sklearn.utils.testing import assert_equal, assert_true
from deep.costs import (SquaredError, BinaryCrossEntropy, NegativeLogLikelihood,
                        PredictionError)


supervised_costs = [NegativeLogLikelihood, PredictionError]
unsupervised_costs = [SquaredError, BinaryCrossEntropy]


class TestCorruptions(unittest.TestCase):

    def test_supervised_cost_call_method(self):
        """Test supervised cost functions."""

        for cost in supervised_costs:
            cost = cost()

            X = T.dmatrix()
            y = T.lvector()

            #: 0d represents the cost function applied to a single prediction
            y_0d = np.empty(1, dtype='int64')

            #: 1d represents the cost function applired to a batch prediction
            y_1d = np.empty(10, dtype='int64')

            if isinstance(cost, PredictionError):
                X_0d = np.empty(1)
                X_1d = np.empty(10)
            else:
                X_0d = np.empty(10)
                X_1d = np.empty((10, 10))

            print cost

            assert_true(isinstance(cost(X, y), T.TensorVariable))
            assert_equal(cost(X_0d, y_0d).ndim, 0)
            assert_equal(cost(X_1d, y_1d).ndim, 0)

    def test_unsupervised_cost_call_methods(self):
        """Test unsupervised cost functions."""

        for cost in unsupervised_costs:
            cost = cost()

            print cost

            X = T.dmatrix()
            y = T.dmatrix()

            #: 1d represents the cost function applied to a single input
            X_1d = y_1d = np.empty(10)

            #: 2d represents the cost function applied to a batch input
            X_2d = y_2d = np.empty((1, 10))

            assert_true(isinstance(cost(X, y), T.TensorVariable))
            assert_equal(cost(X_1d, y_1d).ndim, 0)
            assert_equal(cost(X_2d, y_2d).ndim, 0)

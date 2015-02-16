# -*- coding: utf-8 -*-
"""
    deep.models.base
    ---------------------

    Tests the feed forward neural network model.

    :references: theano deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import unittest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal, assert_equal, assert_raises
from deep.models.nn import NN

iris = load_iris()
X = iris.data
y = iris.target


class TestFeedForwardNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Init and fit FeedForwardNN. """
        cls.clf = NN(layers=[10], n_iter=5)

        from deep.layers import Layer
        from deep.activations import Sigmoid, Softmax

        #: error message for shape mismatch
        #: where to put this?
        layer1 = Layer(size=(4, 100), activation=Sigmoid())
        layer2 = Layer(size=(100, 3), activation=Softmax())

        nn = NN(layers=[layer1, layer2])
        nn.fit(X, y)

        #: fix this
        try:
            cls.clf.fit(X, y)
        except Exception:
            print 'Exception while fitting'

    #: test init

    def test_sklearn_init(self):
        pass

    def test_layerwise_init(self):
        pass

    #: test fit

    def test_fit(self):
        """ Test that costs (% correct) increases each iter of fit. """
        assert_array_equal(np.diff(self.clf._scores) >= 0, True)

    def test_deterministic_fit(self):
        """ Test that multiple fits yield same scores. """
        clf = NN(layers=[10], n_iter=5)
        clf.fit(X, y)
        assert_array_equal(self.clf._scores, clf._scores)

    # :test input dims

    #: how to handle 1D inputs?
    @unittest.SkipTest
    def test_predict_1d(self):
        """ Test predict methods on 1d array.  """
        assert_equal(self.clf.predict_proba(X[0]).shape, iris.target_names.shape)
        assert_equal(self.clf.predict(X[0]).shape, y[0].shape)

    def test_predict_2d(self):
        """ Test predict methods on 2d array.  """
        proba_shape = (2, len(iris.target_names))
        assert_equal(self.clf.predict_proba(X[:2]).shape, proba_shape)
        assert_equal(self.clf.predict(X[:2]).shape, y[:2].shape)

    def test_predict_3d(self):
        """ Test predict methods throw exceptions on 3d array.  """
        assert_raises(ValueError, self.clf.predict_proba, np.ones((1, 1, 1)))
        assert_raises(ValueError, self.clf.predict, np.ones((1, 1, 1)))

import unittest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal, assert_equal, assert_raises
from deep.neural_network.base import FeedForwardNN

iris = load_iris()
X = iris.data
y = iris.target


class TestFeedForwardNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Init and fit FeedForwardNN. """
        cls.clf = FeedForwardNN(layer_sizes=[10], n_iter=5)

        try:
            cls.clf.fit(X, y)
        except Exception:
            print 'Exception while fitting'

    def test_fit(self):
        """ Test that score (% correct) increases each iter of fit. """
        assert_array_equal(np.diff(self.clf._scores) >= 0, True)

    def test_deterministic_fit(self):
        """ Test that multiple fits yield same scores. """
        clf = FeedForwardNN(layer_sizes=[10], n_iter=5)
        clf.fit(X, y)
        assert_array_equal(self.clf._scores, clf._scores)

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

    def test_empty_init_lists(self):
        """ Test __init__ throws exception on empty lists.  """
        assert_raises(ValueError, FeedForwardNN, layer_sizes=[])
        assert_raises(ValueError, FeedForwardNN, activations=[])

    def test_incorrect_activation_type(self):
        """ Test __init__ throws exception for non Layer objects. """
        assert_raises(ValueError, FeedForwardNN, activations=[object])

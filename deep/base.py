import numpy as np
import theano.tensor as T

from abc import ABCMeta
from abc import abstractmethod
from theano import function, config

from deep.costs import PredictionError


class Transformer(object):

    x = T.matrix()

    _transform_function = None

    @abstractmethod
    def transform(self, X):
        """"""

    @abstractmethod
    def _symbolic_transform(self, X):
        """"""

    def fit(self, X):
        return self

    @property
    def params(self):
        return []


class Unsupervised(Transformer):

    _inverse_transform_function = None
    _score_function = None

    def inverse_transform(self, X):
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def score(self, X, y, cost):
        X = np.asarray(X, dtype=config.floatX)
        if not self._score_function or cost != self.cost:
            self._score_function = function([self.x, self.y], self._symbolic_score(self.x, self.y))
        return self._score_function(X, y)

    @abstractmethod
    def fit(self, X):
        """"""

    @abstractmethod
    def _symbolic_inverse_transform(self, X):
        """"""

    @abstractmethod
    def _symbolic_score(self, X, y):
        """"""


class Supervised(object):

    x = T.matrix()
    y = T.lvector()

    _predict_function = None
    _predict_proba_function = None
    _score_function = None

    def predict(self, X):
        if not self._predict_function:
            self._predict_function = function([self.x], self._symbolic_predict(self.x))
        return self._predict_function(X)

    def predict_proba(self, X):
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def score(self, X, y, cost):
        X = np.asarray(X, dtype=config.floatX)
        if not self._score_function or cost != self.cost:
            self._score_function = function([self.x, self.y], self._symbolic_score(self.x, self.y))
        return self._score_function(X, y)

    def _symbolic_predict(self, x):
        return T.argmax(self._symbolic_predict_proba(x), axis=1)

    @abstractmethod
    def _symbolic_predict_proba(self, x):
        """"""

    @abstractmethod
    def _symbolic_score(self, x, y, cost=None):
        if cost is None:
            cost = PredictionError()
        return cost(self._symbolic_predict(x), y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=config.floatX)

        self.fit_method(self, X, y)

        for layer in self.layers:
            layer.corruption = None
        return self

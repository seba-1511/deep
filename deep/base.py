import numpy as np
import theano.tensor as T

from abc import abstractmethod
from theano import function, config

from deep.costs import PredictionError
from deep.datasets.base import SupervisedData


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

#: this base class is basically all the nn code
#: should probably just move this to the network class
class Supervised(object):

    x = T.matrix()
    y = T.lvector()

    _predict_function = None
    _predict_proba_function = None
    _score_function = None

    def predict(self, X):
        #: do we need separate functions for predict and predict_proba?
        #: we could probably just np.argmax here
        #: same comment for score
        if not self._predict_function:
            self._predict_function = function([self.x], self._symbolic_predict(self.x))
        return self._predict_function(X)

    def predict_proba(self, X):
        #: compile these in fit method
        if not self._predict_proba_function:
            self._predict_proba_function = function([self.x], self._symbolic_predict_proba(self.x))
        return self._predict_proba_function(X)

    def score(self, X, y, cost=None):
        X = np.asarray(X, dtype=config.floatX)
        if not self._score_function or cost != self.cost:
            self._score_function = function([self.x, self.y], self._symbolic_score(self.x, self.y))
        return self._score_function(X, y)

    def _symbolic_predict(self, x, noisy=True):
        return T.argmax(self._symbolic_predict_proba(x, noisy), axis=1)

    @abstractmethod
    def _symbolic_predict_proba(self, x, noisy=True):
        """"""

    @abstractmethod
    def _symbolic_score(self, x, y, noisy=True, cost=None):
        if cost is None:
            cost = PredictionError()
        return cost(self._symbolic_predict(x, noisy), y)

    #: should we just remove X, y and take a dataset?
    def fit(self, X, y=None):
        if not isinstance(X, SupervisedData):
            dataset = SupervisedData(X, y)
        else:
            dataset = X

        X = dataset.batch(1)

        for layer in self.layers:
            X = layer.fit_transform(X)

        self.fit_method(self, dataset)

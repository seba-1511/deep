"""
Tied Autoencoder
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import theano.tensor as T

from sklearn.base import BaseEstimator, ClassifierMixin

from theano import function
from theano import shared
from theano.tensor import TensorVariable

from deep.layers.base import Layer
from deep.activations.base import Sigmoid
from deep.fit.base import Fit
from deep.datasets.base import Data
from deep.costs.base import BinaryCrossEntropy
from deep.updates.base import GradientDescent


def handle_dims(predict):
    """  """
    def wrapper(self, X):
        if X.ndim == 1:
            return predict(self, X.reshape(1, -1))[0]
        elif X.ndim == 2:
            return predict(self, X)
        else:
            raise ValueError
    return wrapper



def theano_compatible(func):

    def wrapper(self, X):
        if isinstance(X, TensorVariable):
            return func(self, X)
        else:
            x = T.dmatrix()

            if X.ndim == 1:
                X = X.reshape(1, -1)
                return func(self, x).eval({x: X})[0]
            elif X.ndim == 2:
                return func(self, x).eval({x: X})
            else:
                raise ValueError

    return wrapper

class TiedAE(Layer, BaseEstimator, ClassifierMixin):
    """ """
    def __init__(self, activation=Sigmoid(), learning_rate=10,
                 n_hidden=100, n_iter=10, batch_size=100, _fit=Fit(),
                 _cost=BinaryCrossEntropy(), update=GradientDescent()):

        #: hyperparams (vars)
        self.n_hidden = n_hidden
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size

        #: hyperparams (funcs)
        self._fit = _fit
        self._cost = _cost
        self.update = update

        #: symbolic vars
        self.x = T.dmatrix()
        self.i = T.lscalar()

        #: place holders
        self.data = None
        self._fit_function = None

    @property
    def params(self):
        return self.W, self.b, self.b_decode

    @property
    def updates(self):
        """"""
        rv = list()
        for param in self.params:
            cost = self.score(self.x)
            updates = self.update(cost, param, self.learning_rate)
            for update in updates:
                rv.append(update)
        return rv

    @property
    def givens(self):
        """"""

        #: is there a way to move this to a multilayermodel class?

        X = shared(np.asarray(self.data.X, dtype='float64'))
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size
        return {self.x: X[batch_start:batch_end]}

    @property
    def fit_function(self):
        """"""
        if not self._fit_function:
            self._fit_function = function(inputs=[self.i],
                                          outputs=self.score(self.x),
                                          updates=self.updates,
                                          givens=self.givens)
        return self._fit_function

    @theano_compatible
    def score(self, X):
        return self._cost(self.reconstruct(X), X)

    @theano_compatible
    def transform(self, X):
        return self.activation(T.dot(X, self.W) + self.b)

    @theano_compatible
    def inverse_transform(self, X):
        return self.activation(T.dot(X, self.W.T) + self.b_decode)

    @theano_compatible
    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))

    def fit(self, X, y=None):
        self.data = Data(X)

        size = self.data.features, self.n_hidden
        super(TiedAE, self).__init__(size, self.activation)
        self.b_decode = shared(np.zeros(self.data.features))

        return self._fit(self)


if __name__ == '__main__':
    from deep.datasets.base import load_mnist
    X = load_mnist()[0][0]
    ae = TiedAE(n_iter=1).fit(X)
    print ae.reconstruct(np.random.random((10, 784))).shape
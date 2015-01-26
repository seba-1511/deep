"""
Feed Forward Neural Network
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import theano.tensor as T

from theano import function
from theano import shared
from sklearn.base import BaseEstimator, ClassifierMixin

from deep.layers.base import Layer
from deep.activations.base import Activation
from deep.fit.base import simple_batch_gradient_descent
from deep.hyperparams import layer_sizes, batch_size, n_iter, learning_rate, activations


class FeedForwardNN(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(self, activations=activations, layer_sizes=layer_sizes, learning_rate=learning_rate,
                 n_iter=n_iter, batch_size=batch_size):
        if not activations:
            raise ValueError
        if not layer_sizes:
            raise ValueError

        for activation in activations:
            if not issubclass(activation, Activation):
                raise ValueError

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.layers = []
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    @property
    def params(self):
        """
        """
        return [param for layer in self.layers for param in layer.params]

    def predict_proba(self, X):
        if X.ndim == 1:
            return self._predict_proba(X.reshape(1, -1))[0]
        elif X.ndim == 2:
            return self._predict_proba(X)
        else:
            raise ValueError

    def predict(self, X):
        if X.ndim == 1:
            return self._predict(X.reshape(1, -1))[0]
        elif X.ndim == 2:
            return self._predict(X)
        else:
            raise ValueError

    def _fit(self, X, y):
        """
        """
        i = T.lscalar()
        x = T.dmatrix()
        t = T.lvector()

        feed_forward = lambda input, layer: layer.transform(input)
        predict_proba = reduce(feed_forward, self.layers, x)
        predict = T.argmax(predict_proba, axis=1)

        cost = -T.mean(T.log(predict_proba)[T.arange(t.shape[0]), t])
        score = T.mean(T.eq(predict, t))
        updates = [(param, param - self.learning_rate * T.grad(cost, param))
                   for param in self.params]

        X = shared(np.asarray(X, dtype='float64'))
        y = shared(y)
        start, end = i * self.batch_size, (i+1) * self.batch_size
        givens = {x: X[start: end], t: y[start: end]}

        self._predict = function([x], predict)
        self._predict_proba = function([x], predict_proba)
        self.train = function([i], score, None, updates, givens)

    def fit(self, X, y):
        """
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        n_batches = n_samples / self.batch_size

        if self.layers:
            n_features = self.layers[-1].output_size

        input_sizes = [n_features] + self.layer_sizes
        output_sizes = self.layer_sizes + [n_classes]
        layer_shapes = zip(input_sizes, output_sizes)

        for activation, shape in zip(self.activations, layer_shapes):
            self.layers.append(Layer(shape, activation))

        self._fit(X, y)

        return simple_batch_gradient_descent(self, n_batches)
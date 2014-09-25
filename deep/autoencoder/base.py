from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin

import time
import theano
import theano.tensor as T
import numpy as np


class AutoencoderBase(six.with_metaclass(ABCMeta), BaseEstimator, TransformerMixin):

    @abstractmethod
    def __init__(self, encoder, learning_rate, batch_size,
                 n_iter, verbose):
        self.encoder = encoder
        self.decoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.x = T.dmatrix()

    def fit(self, X, y=None):
        return self._fit(X, y)

    @property
    def params(self):
        return list({self.encoder.W, self.encoder.b_encode,
                     self.decoder.W, self.decoder.b_decode})

    @property
    def updates(self):
        gparams = T.grad(self._score(self.x), self.params)
        return [(param, param - self.learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)]


    def _encode(self, x):
        return self.encoder._encode(x)

    def _decode(self, x):
        return self.decoder._decode(x)

    def _reconstruct(self, x):
        return self._decode(self._encode(x))

    def _score(self, x):
        return T.mean(T.nnet.binary_crossentropy(
            self._reconstruct(x), x))

    def encode(self, x):
        return self._encode(x).eval()

    def decode(self, x):
        return self._decode(x).eval()

    def reconstruct(self, x):
        return self._reconstruct(x).eval()

    def score(self, X):
        return self._score(X).eval()

    def compile(self, X):
        index = T.lscalar()
        begin = index * self.batch_size
        end = begin + self.batch_size
        X_shared = np.asarray(X, dtype=theano.config.floatX)
        X_shared = theano.shared(X_shared, borrow=True)
        return theano.function([index], self._score(self.x), updates=self.updates,
                               givens={self.x: X_shared[begin:end]})

    def _fit(self, X, y=None):
        n_samples, n_features = X.shape
        n_batches = n_samples / self.batch_size

        self.encoder._init_params(n_features)

        score_function = self.compile(X)

        begin = time.time()
        self.scores_ = list()
        for iteration in range(1, self.n_iter + 1):
            score = [score_function(batch_index)
                    for batch_index in range(n_batches)]
            self.scores_.append(np.mean(score))

            if self.verbose:
                end = time.time()
                print("[%s] Iteration %d, score = %.4f,"
                      " time = %.4fs"
                      % (type(self).__name__, iteration,
                         self.scores_[-1], end - begin))
                begin = end

        return self

    def transform(self, X):
        return self.encode(X)


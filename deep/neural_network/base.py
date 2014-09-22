from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.externals import six

import theano.tensor as T
import time


VALID_ENCODERS = ('LinearLayer', 'SigmoidLayer')
VALID_CLASSIFIERS = ('SoftMaxLayer',)


class NeuralNetworkBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    """ Base class for Neural Networks. """

    def __init__(self, encoders):

        assert isinstance(encoders, list)
        assert all(isinstance(encoder, encoders) for encoder in encoders)
        assert len(encoders) >= 1


        self.encoders = encoders

    @abstractmethod
    def fit(self, X, y):
        """ Fit model. """

    def encode(self, x):
        for layer in self.encoders:
            x = layer.encode(x)
        return x


class SupervisedMixin(ClassifierMixin):

    def compile_fit(self, X, y):
        index = T.lscalar()
        begin = index * self.batch_size
        end = begin + self.batch_size
        X_shared = np.asarray(X, dtype=theano.config.floatX)
        X_shared = theano.shared(X_shared, borrow=True)
        y_shared = np.asarray(y, dtype=theano.config.floatX)
        y_shared = T.cast(theano.shared(y_shared, borrow=True), 'int32')
        return theano.function([index], self.cost, updates=self.updates,
                               givens={self.x: X_shared[begin:end],
                                       self.y: y_shared[begin:end]})

    def fit(self, X, y):
        """ Fit the model using X as training data and y as target values. """

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n"
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        n_samples, n_features = X.shape
        n_batches = n_samples / self.batch_size

        cost_function = self.compile_fit(X, y)

        self.input_size = n_features

        begin = time.time()
        self.costs_ = list()
        for iteration in range(1, self.n_iter + 1):
            cost = [cost_function(batch_index)
                    for batch_index in range(n_batches)]
            self.costs_.append(np.mean(cost))

            if self.verbose:
                end = time.time()
                print("[%s] Iteration %d, cost = %.4f,"
                      " time = %.4fs"
                      % (type(self).__name__, iteration,
                         self.costs_[-1], end - begin))
                begin = end

        return self

    def predict(self, x):
        return T.argmax(self.encode(x), axis=1).eval()


class UnsupervisedMixin(TransformerMixin):

    def decode(self, x):
        for layer in self.decoders:
            x = layer.decode(x)
        return x

    def reconstruct(self, inputs):
        return self.decode(self.encode(inputs))

    def compile(self, X):
        index = T.lscalar()
        begin = index * self.batch_size
        end = begin + self.batch_size
        X_shared = np.asarray(X, dtype=theano.config.floatX)
        X_shared = theano.shared(X_shared, borrow=True)
        return theano.function([index], self.cost, updates=self.updates,
                               givens={self.x: X_shared[begin:end]})

    def fit(self, X, y=None):
        """ Fit the model using X as training data. """
        n_samples, n_visible = X.shape
        n_batches = n_samples / self.batch_size

        cost_function = self.compile(X)

        begin = time.time()
        self.costs_ = list()
        for iteration in range(1, self.n_iter + 1):
            cost = [cost_function(batch_index)
                    for batch_index in range(n_batches)]
            self.costs_.append(np.mean(cost))

            if self.verbose:
                end = time.time()
                print("[%s] Iteration %d, cost = %.4f,"
                      " time = %.4fs"
                      % (type(self).__name__, iteration,
                         self.costs_[-1], end - begin))
                begin = end

        return self



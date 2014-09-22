from base import NeuralNetworkBase, SupervisedMixin

import time
import theano
import theano.tensor as T
import numpy as np


class TheanoMultiLayerPerceptron(NeuralNetworkBase, SupervisedMixin):

    def __init__(self, encoders, learning_rate=1, batch_size=10,
                 n_iter=10, verbose=1):
        self.encoders = encoders
        self.x = T.dmatrix('x')
        self.y = T.ivector('y')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose

    @property
    def encoder_params(self):
        return [encoder.W for encoder in self.encoders] + \
               [encoder.b_encode for encoder in self.encoders]

    @property
    def cost(self):
        return -T.mean(T.log(self.encode(self.x))[T.arange(self.y.shape[0]), self.y])

    @property
    def updates(self):
        gparams = T.grad(self.cost, self.encoder_params)
        return [(param, param - self.learning_rate * gparam)
                for param, gparam in zip(self.encoder_params, gparams)]

    def encode(self, x):
        for layer in self.encoders:
            x = layer.encode(x)
        return x

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
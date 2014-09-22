import theano
import theano.tensor as T
import numpy as np
from sklearn.base import BaseEstimator


class Layer(BaseEstimator):

    def __init__(self, size):
        np.random.seed(0)

        self.W = np.asarray(np.random.uniform(
            low=-4 * np.sqrt(6. / sum(size)),
            high=4 * np.sqrt(6. / sum(size)),
            size=size), dtype=theano.config.floatX)
        self.W = theano.shared(self.W, borrow=True)

        self.b_encode = np.zeros(size[1], dtype=theano.config.floatX)
        self.b_encode = theano.shared(self.b_encode, borrow=True)

        self.b_decode = np.zeros(size[0], dtype=theano.config.floatX)
        self.b_decode = theano.shared(self.b_decode, borrow=True)

        self.params = [self.W, self.b_encode, self.b_decode]


class LinearLayer(Layer):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        super(LinearLayer, self).__init__((n_in, n_out))

    def encode(self, x):
        return T.dot(x, self.W) + self.b_encode

    def decode(self, x):
        return T.dot(x, self.W.T) + self.b_decode


class SigmoidLayer(Layer):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        super(SigmoidLayer, self).__init__((n_in, n_out))

    def encode(self, x):
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b_encode)

    def decode(self, x):
        return T.nnet.sigmoid(T.dot(x, self.W.T) + self.b_decode)


class SoftMaxLayer(Layer):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        super(SoftMaxLayer, self).__init__((n_in, n_out))

    def encode(self, x):
        return T.nnet.softmax(T.dot(x, self.W) + self.b_encode)

    def decode(self, x):
        return T.nnet.softmax(T.dot(x, self.W.T) + self.b_decode)
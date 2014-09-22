from base import NeuralNetworkBase, UnsupervisedMixin
from sklearn.neighbors import NearestNeighbors

import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class TheanoStackedAutoEncoder(NeuralNetworkBase, UnsupervisedMixin):
    """ An autoencoder with one or more encoder-decoder pairs. """
    def __init__(self, encoders, decoders=None, learning_rate=1, batch_size=20,
                 n_iter=10, verbose=1):

        self.encoders = encoders

        if decoders:
            self.decoders = decoders
        else:
            self.decoders = self.encoders[::-1]

        self.x = T.dmatrix('x')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose

    @property
    def encoder_params(self):
        return [encoder.W for encoder in self.encoders] + \
               [encoder.b_encode for encoder in self.encoders]

    @property
    def decoder_params(self):
        return [decoder.W for decoder in self.decoders] + \
               [decoder.b_decode for decoder in self.decoders]

    @property
    def params(self):
        return list(set(self.encoder_params + self.decoder_params))

    @property
    def cost(self):
        return T.mean(T.nnet.binary_crossentropy(
            self.reconstruct(self.x), self.x))

    @property
    def updates(self):
        gparams = T.grad(self.cost, self.params)
        return [(param, param - self.learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)]

    def encode(self, x):
        for layer in self.encoders:
            x = layer.encode(x)
        return x

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

    def fit(self, X):
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

    def score(self, X):
        return T.mean(T.nnet.binary_crossentropy(
            self.decode(self.encode(X)), X)).eval()


class TheanoNoisyAutoEncoder(TheanoStackedAutoEncoder):

    def __init__(self, encoders, decoders=None, learning_rate=1, batch_size=20,
                 n_iter=10, verbose=1):

        self.theano_rng = RandomStreams()

        self.encoders = encoders

        if decoders:
            self.decoders = decoders
        else:
            self.decoders = self.encoders[::-1]

        self.x = T.dmatrix('x')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose

    def corrupt(self, x):
        return x + self.theano_rng.normal(size=x.shape, avg=0, std=.1, dtype='float32')

    def reconstruct(self, x):
        return self.decode(self.corrupt(self.encode(x)))


class TheanoNearestNeighborAutoEncoder(object):

    def __init__(self, encoder, decoder=None, learning_rate=1, batch_size=20,
                 n_iter=10, verbose=1):

        self.encoder = encoder

        if decoder:
            self.decoder = decoder
        else:
            self.decoder = encoder

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.theano_rng = RandomStreams()

    @property
    def params(self):
        return [self.encoder.W, self.encoder.b_encode, self.encoder.b_decode]

    def gaussian_noise(self, size):
        return self.theano_rng.normal(size=size, avg=0, std=.1, dtype='float32')

    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, x):
        return self.decoder.decode(x)

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

    def fit(self, X):
        n_samples, n_visible = X.shape
        n_batches = n_samples / self.batch_size

        self.sklearn_neighbors = NearestNeighbors()
        self.sklearn_neighbors.fit(X)
        self.X_neighbors = self.sklearn_neighbors.kneighbors(X, return_distance=False)

        x_neighbors = X[self.X_neighbors[:20]]
        h_neighbors = self.encode(x_neighbors).eval()

        noise = self.gaussian_noise((h_neighbors.shape[0], h_neighbors.shape[2])).eval()




        x_neighbors = x_neighbors.reshape((100, 8, 8))


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


from layers import SigmoidLayer
s1 = SigmoidLayer(700, 500)
s2 = SigmoidLayer(500, 400)
ae = TheanoStackedAutoEncoder([s1, s2])
print ae
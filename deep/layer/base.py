from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.externals import six

import theano
import numpy as np


class LayerBase(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, layer_size):
        self.layer_size = layer_size

    @abstractmethod
    def _encode(self, x):
        """"""

    @abstractmethod
    def _decode(self, x):
        """"""

    def encode(self, x):
        return self._encode(x).eval()

    def decode(self, x):
        return self._decode(x).eval()

    @property
    def params(self):
        return [self.W, self.b_encode, self.b_decode]

    def _init_params(self, layer_input):
        layer_dim = (layer_input, self.layer_size)
        self.W = np.asarray(np.random.uniform(
            low=-4 * np.sqrt(6. / sum(layer_dim)),
            high=4 * np.sqrt(6. / sum(layer_dim)),
            size=layer_dim), dtype=theano.config.floatX)
        self.W = theano.shared(self.W, borrow=True)

        self.b_encode = np.zeros(layer_dim[1], dtype=theano.config.floatX)
        self.b_encode = theano.shared(self.b_encode, borrow=True)

        self.b_decode = np.zeros(layer_dim[0], dtype=theano.config.floatX)
        self.b_decode = theano.shared(self.b_decode, borrow=True)



"""
Layers
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import theano.tensor as T

from deep.activations.base import Activation
from theano import shared


class Layer(object):

    def __init__(self, shape, activation=Activation()):
        np.random.seed(1)
        range = np.sqrt(24. / sum(shape))
        self.W = shared(np.random.uniform(low=-range, high=range, size=shape))
        self.b = shared(np.zeros(shape[1]))
        self.activation = activation.activation

    def transform(self, x):
        """ """
        return self.activation(T.dot(x, self.W) + self.b)

    @property
    def params(self):
        return self.W, self.b


class LinearLayer(Layer):
    """ """
    activation = lambda x: x


class SigmoidLayer(Layer):
    """ """
    activation = T.nnet.sigmoid


class TanhLayer(Layer):
    """ """
    activation = T.tanh


class SoftmaxLayer(Layer):
    """ """
    activation = T.nnet.softmax
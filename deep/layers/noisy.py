"""
Noisy Base Layer and Common Layer Types
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import theano.tensor as T

from deep.layers.base import Layer
from deep.corruptions.base import GaussianCorruption


class Noisy(Layer):

    def __init__(self, shape, corruption=[GaussianCorruption()] * 3):
        super(Noisy, self).__init__(shape)
        self.corrupt = corruption.corrupt

    def transform(self, x):
        corrupt_input = self.corrupt(x)
        corrupt_linear = self.corrupt(T.dot(corrupt_input, self.W) + self.b)
        return self.corrupt(self.activation(corrupt_linear))


class NoisyLinearLayer(Layer):
    """ """
    activation = lambda x: x


class NoisySigmoidLayer(Layer):
    """ """
    activation = T.nnet.sigmoid


class NoisyTanhLayer(Layer):
    """ """
    activation = T.tanh


class NoisySoftmaxLayer(Layer):
    """ """
    activation = T.nnet.softmax
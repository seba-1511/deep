"""
Denoising Base Layer and Common Layer Types
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import theano.tensor as T

from deep.layers.base import Layer
from deep.corruptions.base import GaussianCorruption


class DenoisingLayer(Layer):

    def __init__(self, shape, corruption=GaussianCorruption()):
        super(DenoisingLayer, self).__init__(shape)
        self.corrupt = corruption.corrupt

    def transform(self, x):
        return super(DenoisingLayer, self).transform(self.corrupt(x))


class DenoisingLinearLayer(Layer):
    """ """
    activation = lambda x: x


class DenoisingSigmoidLayer(Layer):
    """ """
    activation = T.nnet.sigmoid


class DenoisingTanhLayer(Layer):
    """ """
    activation = T.tanh


class DenoisingSoftmaxLayer(Layer):
    """ """
    activation = T.nnet.softmax
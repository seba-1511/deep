from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.externals import six
import theano
import theano.tensor as T
import numpy as np

from deep.corruption.base import Corruption
from deep.corruption.base import SaltPepperCorruption


class Layer(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, layer_size):
        self.layer_size = layer_size

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @property
    def params(self):
        return [self.W, self.b_encode, self.b_decode]

    def _init_params(self, layer_dim):

        self.W = np.asarray(np.random.uniform(
            low=-4 * np.sqrt(6. / sum(layer_dim)),
            high=4 * np.sqrt(6. / sum(layer_dim)),
            size=layer_dim), dtype=theano.config.floatX)
        self.W = theano.shared(self.W, borrow=True)

        self.b_encode = np.zeros(layer_dim[1], dtype=theano.config.floatX)
        self.b_encode = theano.shared(self.b_encode, borrow=True)

        self.b_decode = np.zeros(layer_dim[0], dtype=theano.config.floatX)
        self.b_decode = theano.shared(self.b_decode, borrow=True)


class LinearLayer(Layer):

    def __init__(self, layer_size):
        super(LinearLayer, self).__init__(layer_size)

    def encode(self, x):
        return T.dot(x, self.W) + self.b_encode

    def decode(self, x):
        return T.dot(x, self.W.T) + self.b_decode


class NonLinearLayer(LinearLayer):

    def __init__(self, layer_size, activation):
        super(LinearLayer, self).__init__(layer_size)
        self.activation = activation

    def encode(self, x):
        return self.activation(super(NonLinearLayer, self).encode(x))

    def decode(self, x):
        return self.activation(super(NonLinearLayer, self).decode(x))

    def __repr__(self):
        return self.activation.__str__().capitalize() + \
               "Layer(" + str(self.layer_size) +  ")"


class NonLinearDenoisingLayer(NonLinearLayer):

    def __init__(self, layer_size, activation, corruption):
        super(NonLinearDenoisingLayer, self).__init__(layer_size, activation)
        self.corruption = corruption

    def encode(self, x):
        return super(NonLinearDenoisingLayer, self).encode(
            self.corruption.corrupt(x))

    def __repr__(self):
        return "Denoising" +  self.activation.__str__().capitalize() + "Layer" \
               + "(" + str(self.layer_size) + ", " \
               + str(self.corruption.corruption_level) + ")"


class NonLinearNoisyLayer(NonLinearLayer):

    def __init__(self, layer_size, activation, corruption_list):
        super(NonLinearNoisyLayer, self).__init__(layer_size, activation)

        if not len(corruption_list) == 3:
            raise ValueError

        if all([isinstance(c, float) for c in corruption_list]):
            self.corruption_list = [SaltPepperCorruption(corruption_level)
                                    for corruption_level in corruption_list]
        elif all([isinstance(c, Corruption) for c in corruption_list]):
            self.corruption_list = corruption_list
        else:
            raise ValueError

    def encode(self, x):
        return self.corruption_list[2].corrupt(self.activation(
            self.corruption_list[1].corrupt(super(NonLinearLayer, self).encode(
            self.corruption_list[0].corrupt(x)))))

    def __repr__(self):
        corruption_levels = [corruption.corruption_level
                             for corruption in self.corruption_list]
        return corruption.__class__.__name__ +  self.activation.__str__().capitalize() + "Layer" \
       + "(" + str(self.layer_size) + ", " + corruption_levels.__repr__() + ")"


def NonLinearLayerFactory(layer_size, activation, corruption):
    if corruption:
        if isinstance(corruption, float):
            corruption = SaltPepperCorruption(corruption)
            return NonLinearDenoisingLayer(layer_size, activation, corruption)
        elif isinstance(corruption, Corruption):
            return NonLinearDenoisingLayer(layer_size, activation, corruption)
        elif isinstance(corruption, list):
            return NonLinearNoisyLayer(layer_size, activation, corruption)
        else:
            raise ValueError
    else:
        return NonLinearLayer(layer_size, activation)


def SigmoidLayer(layer_size, corruption=None):
    return NonLinearLayerFactory(layer_size, T.nnet.sigmoid, corruption)
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.externals import six

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class Corruption(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, corruption_level):
        self.theano_rng = RandomStreams(0)
        self.corruption_level = corruption_level

    @abstractmethod
    def corrupt(self, x):
        """ Corrupt input. """

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.corruption_level) + ")"


class DummyCorruption(Corruption):

    def corrupt(self, inputs):
        return inputs


class BinomialCorruption(Corruption):

    def corrupt(self, x):
        return x * self.theano_rng.binomial(size=x.shape, n=1,
                                            p=1 - self.corruption_level,
                                            dtype=theano.config.floatX)


class DropoutCorruptor(BinomialCorruption):

    def corrupt(self, x):
        if self.corruption_level < 1e-5:
            return x
        else:
            dropped = super(DropoutCorruptor, self).corrupt(x)
            return dropped * 1.0 / (1.0 - self.corruption_level)


class GaussianCorruption(Corruption):

    def corrupt(self, x):
        return x + self.theano_rng.normal(size=x.shape, avg=0.0,
                                          std=self.corruption_level,
                                          dtype=theano.config.floatX)


class SaltPepperCorruption(Corruption):

    def corrupt(self, x):
        a = self.theano_rng.binomial(size=x.shape, p=(1-self.corruption_level),
                                     dtype=theano.config.floatX)
        b = self.theano_rng.binomial(size=x.shape, p=0.5,
                                     dtype=theano.config.floatX)
        c = T.eq(a, 0) * b
        return x * a + c
from abc import abstractmethod
corruption_level = .5

from theano.tensor.randomstreams import RandomStreams
rng = RandomStreams(1)

import theano.tensor as T


class Corruption(object):

    def __init__(self, corruption_level=corruption_level, rng=rng):
        self.rng = rng
        self.corruption_level = corruption_level

    @abstractmethod
    def corrupt(self, x):
        """"""


class SaltAndPepper(Corruption):

    def corrupt(self, x):
        a = self.rng.binomial(size=x.shape, p=(1-self.corruption_level))
        b = self.rng.binomial(size=x.shape, p=0.5)
        c = T.eq(a, 0) * b
        return x * a + c


class Gaussian(Corruption):

    def corrupt(self, x):
        return x * self.rng.normal(size=x.shape, std=self.corruption_level)


class Binomial(Corruption):

    def corrupt(self, x):
        return x * self.rng.binomial(size=x.shape, p=1 - self.corruption_level)


class Dropout(Binomial):

    def corrupt(self, x):
        if self.corruption_level < 1e-5:
            return x

        dropped = super(Dropout, self).corrupt(x)
        return 1.0 / (1.0 - self.corruption_level) * dropped

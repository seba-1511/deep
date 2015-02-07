# -*- coding: utf-8 -*-
"""
    deep.corruptions.base
    ---------------------

    Implements various types of corruption.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from theano import config

from abc import abstractmethod
from theano.sandbox.rng_mrg import MRG_RandomStreams
from scipy.ndimage.interpolation import rotate, shift, zoom


#: this need documentation that explains the @theano_compatible decorator.


class Corruption(object):
    """An abstract class that represents a corruption function. Once
    initialized, a corruption class corrupts inputs through its
    __call__ method.

    Example::

        corruption_level = 0.5
        corrupt = SaltAndPepper(corruption_level)
        corrupted_input = corrupt(input)

    :param corruption_level: the amount of corruption to add to the input.
    :param rng: a Theano RandomStreams() random number generator.
    """
    def __init__(self, corruption_level=0.5, rng=MRG_RandomStreams(1)):
        self.corruption_level = corruption_level
        self.rng = rng

    @abstractmethod
    def __call__(self, X):
        """"""

    def __repr__(self):
        return str(self.__class__.__name__) + 'Corruption'


class CorruptionSequence(object):
    """An abstract class that represents a corruption function. Once
    initialized, a corruption class corrupts inputs through its
    __call__ method.

    Example::

        corruption_level = 0.5
        corrupt = CorruptionSequence([SaltAndPepper(), Gaussian()])
        corrupted_input = corrupt(input)

    :param corruption_level: the amount of corruption to add to the input.
    :param rng: a Theano RandomStreams() random number generator.
    """

    #: check if all corruptions are theano compatible

    def __init__(self, corruptions):
        self.corruptions = corruptions

    def __call__(self, x):
        for corrupt in self.corruptions:
            x = corrupt(x)
        return x


class Binomial(Corruption):
    """Binomial corruption transforms inputs by changing values to 0, with
    probability 'corruption_level'.

    :param corruption_level: the amount of corruption to add to the input.
    """
    def __call__(self, x):
        return x * self.rng.binomial(size=x.shape, p=1-self.corruption_level, dtype=config.floatX)


class Dropout(Binomial):
    """Dropout corruption transforms inputs by first applying binomial
    corruption and then diving the result by (1 - corruption_level).

    :param corruption_level: the amount of corruption to add to the input.
    """
    def __call__(self, x):
        scaler = 1.0 / (1.0 - self.corruption_level)

        #: why does super()(x) not work?
        return scaler * super(Dropout, self).__call__(x)


class Gaussian(Corruption):
    """Gaussian corruption transforms inputs by adding zero mean isotropic
     Gaussian noise to the values.

    :param corruption_level: the amount of corruption to add to the input.
    """
    def __call__(self, x):
        return x * self.rng.normal(size=x.shape, std=self.corruption_level, dtype=config.floatX)


class SaltAndPepper(Corruption):
    """Salt and pepper corruption transforms inputs by changing values, to
    0 or 1 with equal likelihood, with probability 'corruption_level'.

    :param corruption_level: the amount of corruption to add to the input.
    """
    def __call__(self, X):
        a = self.rng.binomial(size=X.shape, p=1-self.corruption_level, dtype=config.floatX)
        b = self.rng.binomial(size=X.shape, p=0.5, dtype=config.floatX)
        return X * a + T.eq(a, 0) * b


class Rotate(Corruption):

    #: implement in theano

    def __call__(self, X):
        #: is there a cleaner way to do this?
        #:
        #: how do we handle the 1d case?
        val = self.corruption_level
        offset = np.random.uniform(low=-val, high=val)
        angle = (360 + offset) % 360
        return rotate(X, angle, axes=[-1, -2],  reshape=False)


class Shift(Corruption):

    #: implement in theano

    def __call__(self, X):
        #: is there a cleaner way to do this?
        #:
        #: how do we handle the 1d case?
        X = np.asarray(X)
        shifts = np.zeros(X.ndim)
        shift_range = self.corruption_level
        shifts[-2:] = np.random.uniform(low=-shift_range, high=shift_range, size=2)
        return shift(X, shifts)


class Zoom(Corruption):

    #: implement in theano

    def __call__(self, X):
        raise NotImplementedError

        #: is there a cleaner way to do this?
        #:
        #: val = self.corruption_level
        #: offset = np.random.uniform(low=-val, high=val)
        #: zoom_amount = 1 + offset
        #: new_X = zoom(X, zoom_amount)
        #:
        #: how to get center of new array?

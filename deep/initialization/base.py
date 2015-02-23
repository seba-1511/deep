# -*- coding: utf-8 -*-
"""
    deep.initialization.base
    ------------------------

    Implements various initialization methods.

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from abc import abstractmethod
from theano import config, shared


#: is it okay to set seed here?
np.random.seed(1)


class Initialization(object):

    @abstractmethod
    def init_W(self, size):
        raise NotImplementedError

    def init_b(self, size):
        b = np.zeros(size)
        return shared(np.asarray(b, dtype=config.floatX))


class Normal(Initialization):
    """http://arxiv.org/pdf/1409.1556v5.pdf"""

    def __init__(self, scale=.01):
        self.scale = scale

    def init_W(self, size):
        W = np.random.normal(scale=self.scale, size=size)
        return shared(np.asarray(W, dtype=config.floatX))


#: Xavier and MSR are incorrect for conv layers

class Xavier(Initialization):
    """http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf"""

    def init_W(self, size):
        val = 1. / np.sqrt(size[0])
        W = np.random.uniform(low=-val, high=val, size=size)
        return shared(np.asarray(W, dtype=config.floatX))


class MSR(Initialization):
    """http://arxiv.org/pdf/1502.01852v1.pdf"""

    def init_W(self, size):
        val = np.sqrt(2. / size[0])
        W = np.random.normal(loc=0, scale=val, size=size)
        return shared(np.asarray(W, dtype=config.floatX))


class Sparse(Initialization):
    """http://www.icml2010.org/papers/458.pdf"""

    def init_W(self, size):
        raise NotImplementedError

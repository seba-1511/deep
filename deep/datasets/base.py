# -*- coding: utf-8 -*-
"""
    deep.datasets.base
    ---------------------

    Implements dataset base classes

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from theano import config, shared
from abc import abstractmethod




class Dataset(object):
    """"""


#: calling this Supervised conflicts with supervised(model)
class SupervisedDataset(Dataset):

    batch_index = T.lscalar()

    def __init__(self, X, y, augmentation=None):
        self.X_original = X
        self.augmentation = augmentation

        if self.augmentation is not None:
            X = self.augmentation(X)

        self.X = shared(np.asarray(X, dtype=config.floatX))
        self.y = shared(np.asarray(y, dtype='int64'))

    def givens(self, x, y,  batch_size=128):
        batch_start = self.batch_index * batch_size
        batch_end = batch_start + batch_size
        return {x: self.X[batch_start:batch_end],
                y: self.y[batch_start:batch_end]}

    def batch(self, batch_size):
        return self.X.get_value()[:batch_size]

    def update(self):
        if self.augmentation is not None:
            self.X.set_value(self.augmentation(self.X_original))

    def __len__(self):
        return len(self.X.get_value(self.X))
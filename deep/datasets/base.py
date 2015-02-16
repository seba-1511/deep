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


class SupervisedData(object):

    batch_index = T.lscalar()

    def __init__(self, data, augment=None, ):
        X, y = data

        self.X_original = X
        self.augment = augment

        if self.augment is not None:
            for augmentation in self.augment:
                    X = augmentation.transform(X)

        self.X = shared(np.asarray(X, dtype=config.floatX))
        self.y = shared(np.asarray(y, dtype='int64'))

    def givens(self, x, y,  batch_size=128):
        batch_start = self.batch_index * batch_size
        batch_end = (self.batch_index+1) * batch_size
        return {x: self.X[batch_start:batch_end],
                y: self.y[batch_start:batch_end]}

    def batch(self, batch_size):
        return self.X.get_value()[:batch_size]

    def update(self):
        if self.augment is not None:
            for augmentation in self.augment:
                    X = augmentation.transform(self.X_original)
            self.X.set_value(X)

    def __len__(self):
        return len(self.X.get_value(self.X))


class UnsupervisedData(object):

    batch_index = T.lscalar()

    def __init__(self, X, augment=None, ):
        self.X_original = X
        self.augment = augment

        if self.augment is not None:
            for augmentation in self.augment:
                    X = augmentation.transform(X)

        self.X = shared(np.asarray(X, dtype=config.floatX))

    def givens(self, x, batch_size=128):
        batch_start = self.batch_index * batch_size
        batch_end = (self.batch_index+1) * batch_size
        return {x: self.X[batch_start:batch_end]}

    def batch(self, batch_size):
        return self.X.get_value()[:batch_size]

    def update(self):
        if self.augment is not None:
            for augmentation in self.augment:
                    X = augmentation.transform(self.X_original)
            self.X.set_value(X)

    def __len__(self):
        return len(self.X.get_value(self.X))


class SupervisedDataset(object):

    #: normalizes data together

    def __init__(self, dataset, augment):
        self.train = SupervisedData(dataset[0], augment)
        self.valid = SupervisedData(dataset[1], augment)
        self.test = SupervisedData(dataset[2], augment)
        self. augment = augment

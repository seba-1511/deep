# -*- coding: utf-8 -*-
"""
    deep.fit.base
    -------------

    Implements various fitting schemes.

    :references: pylearn2 (cost module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import time
import numpy as np
from abc import abstractmethod


class Fit(object):

    @abstractmethod
    def __call__(self, model):
        """"""


class Iterative(Fit):

    def __init__(self, n_iterations=10, batch_size=100):
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    #: it might be cleaner to pass the data into fit as well and construct
    #: the fit function directly in fit instead of in the model

    def __call__(self, model):

        #: remove this (probably by removing data from model)
        n_batches = model.data.batches(model.batch_size)

        begin = time.time()
        for iteration in range(1, self.n_iterations):
            batch_costs = [model.fit_function(batch) for batch in range(n_batches)]

        print("[%s] Iteration %d, costs = %.2f, time = %.2fs"
              % (type(model).__name__, iteration, np.mean(batch_costs), time.time() - begin))

        return model


class EarlyStopping(Fit):

    def __init__(self, X_valid=None, y_valid=None, n_iterations=10, batch_size=100):
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    def __call__(self, model):

        raise NotImplementedError

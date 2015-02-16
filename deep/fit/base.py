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
import theano.tensor as T
from abc import abstractmethod
from theano import function

from deep.datasets import SupervisedData, UnsupervisedData


class Fit(object):

    def __init__(self, X_valid=None, y_valid=None):
        self.X_valid = X_valid
        self.y_valid = y_valid

    @abstractmethod
    def __call__(self, model, X, y):
        """"""


class Iterative(Fit):

    def __init__(self, n_iterations=10, batch_size=100, valid=None):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.valid = valid

    def __call__(self, model, dataset):
        x = T.matrix()
        index = [dataset.batch_index] #: pull this out of dataset

        if isinstance(dataset, SupervisedData):
            y = T.lvector()
            train_score = model._symbolic_score(x, y)
            updates = model.updates(x, y)
            train_givens = dataset.givens(x, y, self.batch_size)

            if self.valid is not None:

                #: remove noise to compile clean score
                for layer in model.layers:
                    layer.corruption = None

                valid_score = model._symbolic_score(x, y)
                valid_givens = self.valid.givens(x, y, self.batch_size)

        if isinstance(dataset, UnsupervisedData):
            train_score = model._symbolic_score(x)
            updates = model.updates(x)
            train_givens = dataset.givens(x, self.batch_size)

            if self.valid is not None:
                valid_score = model._symbolic_score(x)
                valid_givens = self.valid.givens(x, self.batch_size)

        train = function(index, train_score, None, updates, train_givens)
        n_train_batches = len(dataset) / self.batch_size

        if self.valid is not None:
            valid = function(index, valid_score, None, None, valid_givens)
            n_valid_batches = len(self.valid) / self.batch_size
        else:
            n_valid_batches = 0

        for iteration in range(1, self.n_iterations + 1):
            begin = time.time()
            train_costs = [train(batch) for batch in range(n_train_batches)]
            valid_costs = [valid(batch) for batch in range(n_valid_batches)]
            elapsed = time.time() - begin

            train_cost = np.mean(train_costs)
            valid_cost = 0
            if self.valid is not None:
                valid_cost = np.mean(valid_costs)

            print("[%s] Iteration %d, train = %.2f, valid = %.4f, time = %.2fs"
                  % (type(model).__name__, iteration, train_cost, valid_cost, elapsed))

            dataset.update()

        return model


class EarlyStopping(Fit):

    #: how to combine this with iterative fit?
    #: if valid is none, stop based on train

    def __init__(self, valid, n_iterations=100, batch_size=100):
        self.valid = valid
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.i = T.lscalar()

    def __call__(self, model, dataset):
        x = model.x
        y = model.y
        index = [dataset.batch_index]

        from deep.costs import NegativeLogLikelihood

        score = NegativeLogLikelihood()(model._symbolic_predict_proba(x), y)
        updates = model.updates
        givens = dataset.givens(x, y, self.batch_size)
        train = function(index, score, None, updates, givens)

        score = NegativeLogLikelihood()(model._symbolic_predict_proba(x, noisy=False), y)
        givens = self.valid.givens(x, y, self.batch_size)
        valid = function(index, score, None, None, givens)

        n_train_batches = len(dataset) / self.batch_size
        n_valid_batches = len(self.valid) / self.batch_size

        last_valid_cost = 100
        for iteration in range(1, self.n_iterations + 1):
            begin = time.time()
            train_costs = [train(batch) for batch in range(n_train_batches)]
            valid_costs = [valid(batch) for batch in range(n_valid_batches)]
            elapsed = time.time() - begin

            train_cost = np.mean(train_costs)
            valid_cost = np.mean(valid_costs)

            if valid_cost > last_valid_cost:
                break
            else:
                last_valid_cost = valid_cost

            print("[%s] Iteration %d, train = %.2f, valid = %.2f, time = %.2fs"
                  % (type(model).__name__, iteration, train_cost, valid_cost, elapsed))

            dataset.update()

        return model

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
from theano import shared, config, function


class Fit(object):

    def __init__(self, X_valid, y_valid):
        self.X_valid = X_valid
        self.y_valid = y_valid

    @abstractmethod
    def __call__(self, model, X, y):
        """"""


class Iterative(Fit):

    def __init__(self, n_iterations=10, batch_size=100, augmentation=None):
        self.n_iterations = n_iterations
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.i = T.lscalar()

    #: it might be cleaner to pass the data into fit as well and construct
    #: the fit function directly in fit instead of in the model

    def __call__(self, model, X, y):
        n_batches = len(X) / self.batch_size
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size

        if self.augmentation is not None:
            X_augmented = self.augmentation(X)
            model._fit(X_augmented, y)
            X_shared = shared(np.asarray(X_augmented, dtype=config.floatX))
        else:
            model._fit(X, y)
            X_shared = shared(np.asarray(X, dtype=config.floatX))

        y_shared = shared(np.asarray(y, dtype='int64'))
        givens = {model.x: X_shared[batch_start:batch_end],
                  model.y: y_shared[batch_start:batch_end]}

        fit_function = function(inputs=[self.i],
                                outputs=model._symbolic_score(model.x, model.y),
                                updates=model.updates,
                                givens=givens)

        begin = time.time()

        for iteration in range(1, self.n_iterations + 1):

            batch_costs = [fit_function(batch) for batch in range(n_batches)]

            print("[%s] Iteration %d, costs = %.2f, time = %.2fs"
                  % (type(model).__name__, iteration, np.mean(batch_costs), time.time() - begin))

            if self.augmentation is not None:
                X_shared.set_value(self.augmentation(X))

        return model


class EarlyStopping(Fit):

    def __init__(self, X_valid=None, y_valid=None, n_iterations=10, batch_size=100):
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    def __call__(self, model):

        raise NotImplementedError


class Joint(Fit):

    def __init__(self, X_valid=None, y_valid=None, augmentation=None):
        super(Joint, self).__init__(X_valid, y_valid)
        self.augmentation = augmentation
        self.i = T.lscalar()

    def __call__(self, models, X, y):

        first_model = models[0]

        self.batch_size = first_model.fit_method.batch_size
        self.n_iterations = first_model.fit_method.n_iterations

        n_batches = len(X) / self.batch_size
        batch_start = self.i * self.batch_size
        batch_end = (self.i+1) * self.batch_size

        if self.augmentation is not None:
            X_augmented = self.augmentation(X)
            for model in models:
                model._fit(X_augmented, y)
            X_shared = shared(np.asarray(X_augmented, dtype=config.floatX))
        else:
            for model in models:
                model._fit(X, y)
            X_shared = shared(np.asarray(X, dtype=config.floatX))

        y_shared = shared(np.asarray(y, dtype='int64'))
        givens = {first_model.x: X_shared[batch_start:batch_end],
                  first_model.y: y_shared[batch_start:batch_end]}

        score = 0
        params = []
        for model in models:
            score += model._symbolic_cost(first_model.x, first_model.y)
            params.extend(model.params)

        params = list(set(params))

        updates = []
        for param in params:
            updates.extend(first_model.update(score, param, first_model.learning_rate))

        fit_function = function(inputs=[self.i],
                                outputs=first_model._symbolic_score(first_model.x, first_model.y),
                                updates=updates,
                                givens=givens)

        begin = time.time()

        for iteration in range(1, self.n_iterations + 1):

            batch_costs = [fit_function(batch) for batch in range(n_batches)]

            print("[%s] Iteration %d, costs = %.2f, time = %.2fs"
                  % (type(model).__name__, iteration, np.mean(batch_costs), time.time() - begin))

            for model in models:
                print model.score(self.X_valid, self.y_valid)

            if self.augmentation is not None:
                X_shared.set_value(self.augmentation(X))

        return model

# -*- coding: utf-8 -*-
"""
    deep.fit.base
    -------------

    Implements various fitting schemes.

    :references: nolearn

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import time
import numpy as np
import theano.tensor as T
from theano import function, shared, config
from deep.augmentation import Augmentation


def _print_header():
    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(iteration, train_cost, valid_cost, elapsed):
    print("  {:>5} | {:>7.4f} | {:>7.4f} | {:>4.1f}s".format(
        iteration, train_cost, valid_cost, elapsed))


#: separate X, y givens to combine these
def supervised_givens(i, x, X, y, Y, batch_size):
    X = shared(np.asarray(X, dtype=config.floatX))
    Y = shared(np.asarray(Y, dtype='int64'))
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    return {x: X[batch_start:batch_end],
            y: Y[batch_start:batch_end]}


def unsupervised_givens(i, x, X, batch_size):
    X = shared(np.asarray(X, dtype=config.floatX))
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    return {x: X[batch_start:batch_end]}


class Fit(object):

    """
        The function that defines the interface for the fit functions
    """

    def __init__(self):
        raise NotImplementedError("Should have implemented this")

    def fit(self):
        raise NotImplementedError("Should have implemented this")

    def finished(self):
        raise NotImplementedError("Should have implemented this")


class Iterative(Fit):

    def __init__(self, n_iterations=100, batch_size=128, valid_size=0.1,
                 save=False, augmentation=Augmentation):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.train_scores = [np.inf]
        self.valid_scores = [np.inf]
        self.save = save

    #: does it matter that these are class variables?
    x = T.matrix()
    y = T.lvector()
    i = T.lscalar()

    def save_best(self, model, score):
        if self.save and score < self.valid_scores[-1]:
            model.save()

    def augment_data(self, X):
        pass

    def compile_train_function(self, model, X, y):
        if y is None:
            score = model._symbolic_score(self.x)
            updates = model.updates(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)
        else:
            #: for plankton competition
            from deep.costs import NegativeLogLikelihood
            score = NegativeLogLikelihood()(
                model._symbolic_predict_proba(self.x), self.y)
            # score = model._symbolic_score(self.x, self.y)

            updates = model.updates(self.x, self.y)
            givens = supervised_givens(
                self.i, self.x, X, self.y, y, self.batch_size)
        return function([self.i], score, None, updates, givens)

    def compile_valid_function(self, model, X, y):
        if y is None:
            score = model._symbolic_score(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)
        else:
            #: hacky dropout fix to get clean valid predictions
            for layer in model.layers:
                layer.corruption = None

            #: for plankton competition
            from deep.costs import NegativeLogLikelihood
            score = NegativeLogLikelihood()(
                model._symbolic_predict_proba(self.x), self.y)
            givens = supervised_givens(
                self.i, self.x, X, self.y, y, self.batch_size)
        return function([self.i], score, None, None, givens)

    def fit(self, model, X, y=None, X_valid=None, y_valid=None):
        n_train_batches = len(X) / self.batch_size
        train_function = self.compile_train_function(model, X, y)

        if X_valid is not None:
            n_valid_batches = len(X_valid) / self.batch_size
            valid_function = self.compile_valid_function(
                model, X_valid, y_valid)

        _print_header()

        for iteration in xrange(1, self.n_iterations + 1):
            begin = time.time()
            train_costs = [train_function(batch)
                           for batch in xrange(n_train_batches)]
            train_cost = np.mean(train_costs)
            self.train_scores.append(train_cost)

            valid_cost = np.inf
            if X_valid is not None:
                valid_costs = [valid_function(batch)
                               for batch in xrange(n_valid_batches)]
                valid_cost = np.mean(valid_costs)
                self.save_best(model, valid_cost)
            self.valid_scores.append(valid_cost)
            elapsed = time.time() - begin

            _print_iter(iteration, train_cost, valid_cost, elapsed)

            if self.finished:
                break

        return model

    @property
    def finished(self):
        return False

    def __str__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.n_iterations, self.batch_size)


class EarlyStopping(Iterative):

    def __init__(self, patience=1, n_iterations=100, batch_size=128):
        super(EarlyStopping, self).__init__(n_iterations, batch_size)
        self.patience = patience

    @property
    def finished(self):
        if len(self.valid_scores) <= self.patience:
            return False
        else:
            return self.valid_scores[-1] > self.valid_scores[-(self.patience + 1)]

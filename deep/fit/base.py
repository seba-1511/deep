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

from sklearn.cross_validation import train_test_split


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
    batch_end = (i+1) * batch_size
    return {x: X[batch_start:batch_end],
            y: Y[batch_start:batch_end]}


def unsupervised_givens(i, x, X, batch_size):
    X = shared(np.asarray(X, dtype=config.floatX))
    batch_start = i * batch_size
    batch_end = (i+1) * batch_size
    return {x: X[batch_start:batch_end]}


class Iterative(object):

    def __init__(self, n_iterations=100, batch_size=128, valid_size=0.1):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.valid_size = valid_size

    x = T.matrix()
    y = T.lvector()
    i = T.lscalar()

    #: sign flipped for plankton
    #: how to handle init in general case?
    train_scores = [np.inf]
    valid_scores = [np.inf]

    def compile_train_function(self, model, X, y):
        if y is None:
            score = model._symbolic_score(self.x)
            updates = model.updates(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)

        else:

            #: for plankton competition
            from deep.costs import NegativeLogLikelihood
            score = NegativeLogLikelihood()(model._symbolic_predict_proba(self.x), self.y)
            #score = model._symbolic_score(self.x, self.y)

            updates = model.updates(self.x, self.y)
            givens = supervised_givens(self.i, self.x, X, self.y, y, self.batch_size)
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
            score = NegativeLogLikelihood()(model._symbolic_predict_proba(self.x), self.y)
            #score = model._symbolic_score(self.x, self.y)

            givens = supervised_givens(self.i, self.x, X, self.y, y, self.batch_size)
        return function([self.i], score, None, None, givens)

    def fit(self, model, X, y=None):
        if y is None:
            X_train, X_valid = train_test_split(X, test_size=self.valid_size)
            y_train, y_valid = None, None
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)

        n_train_batches = len(X_train) / self.batch_size
        n_valid_batches = len(X_valid) / self.batch_size

        train_function = self.compile_train_function(model, X_train, y_train)
        valid_function = self.compile_valid_function(model, X_valid, y_valid)

        _print_header()

        for iteration in range(1, self.n_iterations+1):
            begin = time.time()

            train_costs = [train_function(batch) for batch in range(n_train_batches)]
            valid_costs = [valid_function(batch) for batch in range(n_valid_batches)]

            train_cost = np.mean(train_costs)
            valid_cost = np.mean(valid_costs)

            self.train_scores.append(train_cost)
            self.valid_scores.append(valid_cost)

            elapsed = time.time() - begin

            _print_iter(iteration, train_cost, valid_cost, elapsed)

            if self.finished:
                break

        return model

    @property
    def finished(self):
        return False


class EarlyStopping(Iterative):

    @property
    def finished(self):
        #: sign flipped for plankton
        #: need to add a parameter to costs that specifies
        #: whether it is a increasing or decreasing cost.
        return self.valid_scores[-1] > self.valid_scores[-2]


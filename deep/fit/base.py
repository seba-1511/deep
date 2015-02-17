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

class ansi:
    GREEN = '\033[32m'
    ENDC = '\033[0m'
ansi = ansi()


def _print_header():
    print("""
  Epoch |  Train  |  Valid  |  Time
--------|---------|---------|--------\
""")


def _print_iter(best_train, best_valid, elapsed, iteration, train_cost, valid_cost):
    print("  {:>5} | {}{:>7.4f}{} | {}{:>7.4f}{} "
          "| {:>4.1f}s".format(
        iteration,
        ansi.GREEN if best_train else "",
        train_cost,
        ansi.ENDC if best_train else "",
        ansi.GREEN if best_valid else "",
        valid_cost,
        ansi.ENDC if best_valid else "",
        elapsed,
    ))



#: separate X, y givens to combine these
def supervised_givens(i, x, X, y, Y, batch_size):
    raise NotImplementedError


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

    def compile_train_function(self, model, X, y):
        if y is None:
            score = model._symbolic_score(self.x)
            updates = model.updates(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)

        else:
            score = model._symbolic_score(self.x, self.y)
            updates = model.updates(self.x, self.y)
            givens = supervised_givens(self.i, self.x, X, self.y, y, self.batch_size)
        return function([self.i], score, None, updates, givens)

    def compile_valid_function(self, model, X, y):
        if y is None:
            score = model._symbolic_score(self.x)
            givens = unsupervised_givens(self.i, self.x, X, self.batch_size)
        else:
            score = model._symbolic_score(self.x, self.y)
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

        best_train_cost = np.inf
        best_valid_cost = np.inf

        for iteration in range(1, self.n_iterations+1):
            begin = time.time()

            train_costs = [train_function(batch) for batch in range(n_train_batches)]
            valid_costs = [valid_function(batch) for batch in range(n_valid_batches)]

            train_cost = np.mean(train_costs)
            valid_cost = np.mean(valid_costs)

            if train_cost < best_train_cost:
                best_train_cost = train_cost
            if valid_cost < best_valid_cost:
                best_valid_cost = valid_cost

            best_train = best_train_cost == train_cost
            best_valid = best_valid_cost == valid_cost

            elapsed = time.time() - begin

            _print_iter(best_train, best_valid, elapsed,
                        iteration, train_cost, valid_cost)

        return model
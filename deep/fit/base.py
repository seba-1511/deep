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

#: this should be moved the model's __init__ method.
verbose = True


class Fit(object):
    """An abstract class that represents a fit function. Once initialized, a
    fit class returns a fitted model through its __call__ method.

    :param model:
    :return:
    """

    #: this class needs a better name

    def __init__(self):
        pass

    def __call__(self, model):
        n_batches = model.data.batches(model.batch_size)
        begin = time.time()
        model._scores = []

        for iter in range(1, model.n_iter+1):
            batch_costs = []
            for batch in range(n_batches):
                batch_costs.append(model.fit_function(batch))
            model._scores.append(np.mean(batch_costs))

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, costs = %.2f, time = %.2fs"
                      % (type(model).__name__, iter, model._scores[-1], end - begin))
                begin = end

        return model

    def __repr__(self):
        return str(self.__class__.__name__)

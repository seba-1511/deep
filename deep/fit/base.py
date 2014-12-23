import time
import numpy as np

verbose = True

def simple_batch_gradient_descent(model, n_batches):

    begin = time.time()
    model._scores = []

    for iter in range(1, model.n_iter+1):
        batch_costs = []
        for batch in range(n_batches):
            batch_costs.append(model.train(batch))
        model._scores.append(np.mean(batch_costs))

        if verbose:
            end = time.time()
            print("[%s] Iteration %d, score = %.2f, time = %.2fs"
                  % (type(model).__name__, iter, model._scores[-1], end - begin))
            begin = end

    return model
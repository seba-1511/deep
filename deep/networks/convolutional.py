# -*- coding: utf-8 -*-
"""
    deep.networks.convolutional
    ---------------------------

    Implements a convolutional neural network.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

__author__ = 'Gabriel Pereyra <gbrl.pereyra@gmail.com>'

__license__ = 'BSD 3 clause'

from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.layers.base import ConvolutionLayer
from deep.updates.base import GradientDescent
from deep.datasets.base import Data
from deep.networks.base import FeedForwardNN
from deep.activations.base import Sigmoid, Softmax
from deep.utils.base import theano_compatible, reshape_1d
from deep.fit.base import Fit


class ConvolutionalNN(FeedForwardNN):
    """ """
    def __init__(self, n_filters=(2, 50), filter_sizes=(5, 5), pool_sizes=(2, 2), layer_sizes=(100,),
                 n_iter=100, batch_size=100, learning_rate=.1,
                 activations=(Sigmoid(), Sigmoid(), Softmax()),
                 cost=NegativeLogLikelihood(), update=GradientDescent(),
                 score=PredictionError(), fit=Fit()):
        self.n_filters = n_filters
        self.pool_sizes = pool_sizes
        self.filter_sizes = filter_sizes
        self.conv_layers = []
        super(ConvolutionalNN, self).__init__(activations, layer_sizes,
                                              learning_rate, n_iter, batch_size,
                                              cost, update, fit, score)

    #@theano_compatible
    #def predict_proba(self, X):


    @theano_compatible
    @reshape_1d
    def predict_proba(self, X):
        import numpy as np
        n_dims = int(np.sqrt(self.data.features))

        x = X.reshape((self.batch_size, 1, n_dims, n_dims))

        for layer in self.conv_layers:
            x = layer.__call__(x)
        x = x.flatten(2)

        for layer in self.layers:
            x = layer.__call__(x)

        return x

    def fit(self, X, y):
        """ """
        self.data = Data(X, y)

        import numpy as np
        dim = int(np.sqrt(self.data.features))

        self.activations = list(self.activations)
        prev_filters = 1
        conv_shape = zip(self.n_filters, self.pool_sizes, self.filter_sizes)
        for n_filters, pool, filter_size in conv_shape:
            size = (n_filters, prev_filters, filter_size, filter_size)
            activation = self.activations.pop(0)
            self.conv_layers.append(ConvolutionLayer(size, pool, activation))
            prev_filters = n_filters
            dim = self.conv_layers[-1].output_dim(dim)

        self.data.features = self.conv_layers[-1].output_shape(dim)

        return super(ConvolutionalNN, self).fit(X, y)

if __name__ == '__main__':

    import numpy as np

    from deep.datasets import load_mnist
    X, y = load_mnist()[0]
    clf = ConvolutionalNN(n_iter=1).fit(X, y)



    print clf.predict_proba(np.random.random((10, 784)))

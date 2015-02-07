# -*- coding: utf-8 -*-
"""
    deep.networks.convolutional
    ---------------------------

    Implements a convolutional neural network.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from theano import config

from deep.costs.base import NegativeLogLikelihood, PredictionError
from deep.layers.base import ConvolutionLayer
from deep.updates.base import GradientDescent
from deep.datasets.base import Data
from deep.networks.base import FeedForwardNN
from deep.activations.base import Sigmoid, Softmax
from deep.fit.base import Fit
from deep.layers import Layer


class ConvolutionalNN(FeedForwardNN):
    """ """
    def __init__(self, n_filters=(2, 50), filter_size=5, pool_size=2, layer_sizes=(100,),
                 n_iter=100, batch_size=100, learning_rate=.1, activation=Sigmoid(),
                 _cost=NegativeLogLikelihood(), update=GradientDescent(),
                 _score=PredictionError(), _fit=Fit(), corruption=None):
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.filter_size = filter_size
        self.conv_layers = []
        super(ConvolutionalNN, self).__init__(layer_sizes, activation,
                                              learning_rate, n_iter, batch_size,
                                              _cost, update, _fit, _score, corruption)

    def _symbolic_predict_proba(self, X):
        x = X.reshape(self._input_shape)

        #: combines these layers into self
        for layer in self.conv_layers:
            x = layer._symbolic_transform(x)

        x = x.flatten(2)
        for layer in self.layers:
            x = layer._symbolic_transform(x)
        return x

    @property
    def _input_shape(self):
        #: Should we make this a network property that specifies input shape
        #: Also for layers as well? (then this just returns the first layer param)
        n_dims = int(np.sqrt(self.data.features))
        return self.batch_size, 1, n_dims, n_dims

    def fit(self, X, y):
        """ """
        #: remove data (just store x, y directly?)
        self.data = Data(X, y)

        #: better name for dummy batch
        dummy_batch = np.zeros(self._input_shape, dtype=config.floatX)

        #: init conv layers
        for n_filters in self.n_filters:
            size = (n_filters, dummy_batch.shape[1], self.filter_size, self.filter_size)
            layer = ConvolutionLayer(size, self.pool_size, self.activation)
            self.conv_layers.append(layer)
            dummy_batch = layer.transform(dummy_batch)

        dummy_batch = dummy_batch.reshape(self.batch_size, -1)

        #: init layers
        for layer_size in self.layer_sizes:
            size = (dummy_batch.shape[1], layer_size)
            layer = Layer(size, self.activation)
            self.layers.append(layer)
            dummy_batch = layer.transform(dummy_batch)

        #: init softmax layer
        size = (dummy_batch.shape[1], self.data.classes)
        self.layers.append(Layer(size, Softmax()))

        self._fit(self)

        #: hack to get clean predictions after training
        #: this fails if we retrain the model since it won't
        #: have the original corruption.
        for layer in self:
            layer.corruption = None
        return self

# -*- coding: utf-8 -*-
"""
    deep.regularizers.base
    -------------------------

    Implements various types of regularization.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

from abc import abstractmethod

import theano.tensor as T

class Regularizer(object):

    def __init__(self, alpha=.00001):
        self.alpha = alpha

    @abstractmethod
    def __call__(self, X):
        pass

    def __repr__(self):
        return str(self.__class__.__name__) + 'Regularizer'


class L1(Regularizer):

    def __call__(self, param):
        return self.alpha * T.sum(abs(param))


class L2(Regularizer):

    def __call__(self, param):
        return self.alpha * T.sum(param ** 2)


import numpy as np
from deep.layers import Layer
from deep.activations import Softmax


class Supervised(Regularizer):

    def __init__(self, classifier, alpha=.5):
        self.classifier = classifier
        super(Supervised, self).__init__(alpha)

    #: this only updates layer with respect to aux cost
    #: how to propagate this gradient to lower levels as well?
    def __call__(self, layer):
        self.auxiliary_head = Layer(self.classifier.data.classes, Softmax())
        self.auxiliary_head.fit(np.zeros(layer.shape))

        x = self.classifier.x
        for layer in self.classifier:
            x = layer._symbolic_transform(x)
            if layer == layer:
                break
        x = self.auxiliary_head._symbolic_transform(x)
        return self.alpha * self.classifier._cost(x, self.classifier.y)
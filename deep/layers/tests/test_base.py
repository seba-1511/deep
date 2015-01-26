# -*- coding: utf-8 -*-
"""
    deep.layers.test.test_base
    --------------------------

    Tests the different layer classes.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import unittest
import theano.tensor as T
import numpy as np

from sklearn.utils.testing import assert_equal, assert_true
from deep.layers import Layer, ConvolutionLayer, DenoisingLayer

layers = [Layer, DenoisingLayer, ConvolutionLayer]


class TestLayers(unittest.TestCase):

    def test_layer_call_method(self):

        for layer in layers:
            layer = layer()
            shape = layer.shape

            #: how to combine assert logic?
            #: how to handle single case for conv layer?

            #: conv layer takes 2D (single) or 4D (batch)
            if isinstance(layer, ConvolutionLayer):
                X = T.tensor4()
                X_batch = np.empty((1, 1, 10, 10))

                #: where to put func to calculate conv output size?
                assert_equal(layer(X_batch).shape, (1, 10, 3, 3))

            #: other layers take 1D (single) or 4D (batch)
            else:
                X = T.dmatrix()
                X_single = np.empty(shape[0])
                X_batch = np.empty((1, shape[0]))

                #: check that layer handles single and batch inputs
                assert_equal(layer(X_single).shape, (shape[1], ))
                assert_equal(layer(X_batch).shape, (1, shape[1]))

            #: check if layer is theano compatible
            assert_true(isinstance(layer(X), T.TensorVariable))

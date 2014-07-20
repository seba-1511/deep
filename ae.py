"""
Autoencoder
"""

import numpy as np
from mlp import MLP
import mlp

class AE(MLP):
    """ a basic autoencoder """

    def __init__(self, layers):
        """ initialize with two layers """

        # TODO: better initialization. One layer and reverse?

        assert layers[0].input_size == layers[1].output_size

        # TODO: change to super()
        self.layers = layers



layer1 = mlp.SigmoidLayer(784, 30)
layer2 = mlp.SigmoidLayer(30, 10)
layers = [layer1, layer2]

#ae = AE(layers)
mlp = MLP(layers)

import train
import load_data

data = load_data.mnist()
data = load_data.reshape(data)

train.sgd(data, mlp)
"""
Autoencoder
"""

from deep.model.mlp import MLP


class AE(MLP):
    """ a basic autoencoder """

    def __init__(self, layers):
        """ initialize with two layers """

        # TODO: better initialization. One layer and reverse?

        assert layers[0].input_size == layers[1].output_size

        # TODO: change to super()
        self.layers = layers

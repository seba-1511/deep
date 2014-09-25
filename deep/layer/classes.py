import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

from deep.layer.base import LayerBase
from deep.corruption.base import Corruption
from deep.corruption.base import NoCorruption
from deep.corruption.base import SaltPepperCorruption


class LinearLayerBase(LayerBase):

    def __init__(self, layer_size=10,
                 input_corruption=NoCorruption(),
                 output_corruption=NoCorruption()):
        super(LinearLayerBase, self).__init__(layer_size)
        self.input_corruption = input_corruption
        self.output_corruption = output_corruption

    def _encode(self, x):
        x = self.input_corruption.corrupt(x)
        x = T.dot(x, self.W) + self.b_encode
        x = self.output_corruption.corrupt(x)
        return x

    def _decode(self, x):
        return T.dot(x, self.W.T) + self.b_decode


class NonLinearLayerBase(LinearLayerBase):

    def __init__(self, layer_size=10,
                 activation=sigmoid,
                 input_corruption=NoCorruption(),
                 pre_corruption=NoCorruption(),
                 post_corruption=NoCorruption()):
        super(NonLinearLayerBase, self).__init__(layer_size,
                                                 input_corruption,
                                                 pre_corruption)
        self.activation = activation
        self.post_corruption = post_corruption

    def _encode(self, x):
        x = super(NonLinearLayerBase, self)._encode(x)
        x = self.activation(x)
        x = self.post_corruption.corrupt(x)
        return x

    def _decode(self, x):
        linear = super(NonLinearLayerBase, self)._decode(x)
        return self.activation(linear)


def initialize_corruption(corruption):
    if not corruption:
        return NoCorruption()
    if isinstance(corruption, float):
        if corruption > 0 and corruption < 1:
            return SaltPepperCorruption(corruption)
        else:
            raise ValueError
    else:
        raise ValueError


def NonLinearFactory(layer_size, activation, input_corruption,
                     pre_corruption, post_corruption):
    input_corruption = initialize_corruption(input_corruption)
    pre_corruption = initialize_corruption(pre_corruption)
    post_corruption = initialize_corruption(post_corruption)
    return NonLinearLayerBase(layer_size, activation, input_corruption,
                              pre_corruption, post_corruption)


def LinearLayer(layer_size=10, input_corruption=None, output_corruption=None):
    input_corruption = initialize_corruption(input_corruption)
    output_corruption = initialize_corruption(output_corruption)
    return LinearLayerBase(layer_size, input_corruption, output_corruption)


def SigmoidLayer(layer_size=10, input_corruption=None, pre_corruption=None,
                 post_corruption=None):
    return NonLinearFactory(layer_size, sigmoid, input_corruption,
                            pre_corruption, post_corruption)

def TanhLayer(layer_size=10, input_corruption=None, pre_corruption=None,
                 post_corruption=None):
    return NonLinearFactory(layer_size, tanh, input_corruption,
                            pre_corruption, post_corruption)

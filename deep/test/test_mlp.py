from deep.model.mlp import sigmoid
from deep.model.mlp import Layer
from deep.model.mlp import LinearLayer
from deep.model.mlp import SigmoidLayer
from deep.model.mlp import MLP
import numpy as np


def test_layer():
    """ testing layer """

    # check if activations below are saved
    layer = Layer(10, 10)
    layer.fprop(np.arange(10))

    assert np.all(layer.activation_below == np.arange(10))


def test_linear_layer():
    """ testing linear layer """

    # check if weight shapes are initialized correctly
    linear_layer = LinearLayer(10, 1)
    assert linear_layer.weights.shape == (1, 10)
    assert linear_layer.bias.shape == (1, 1)

    # check if linear fprop works with identity matrix as weights
    linear_layer.weights = np.eye(10)
    assert np.all(linear_layer.fprop(np.arange(10)) == np.arange(10))

    # check if activations are stored
    assert np.all(linear_layer.activation_below == np.arange(10))
    assert np.all(linear_layer.activation_linear == np.arange(10))


def test_sigmoid_layer():
    """ testing sigmoid layer """

    # check if weight shapes are initialized correctly
    sigmoid_layer = SigmoidLayer(10, 1)
    assert sigmoid_layer.weights.shape == (1, 10)
    assert sigmoid_layer.bias.shape == (1, 1)

    # check if sigmoid fprop works with identity matrix as weights
    sigmoid_layer.weights = np.eye(10)
    assert np.all(sigmoid_layer.fprop(np.arange(10)) == sigmoid(np.arange(10)))

    # check if activations are stored
    assert np.all(sigmoid_layer.activation_below == np.arange(10))
    assert np.all(sigmoid_layer.activation_linear == np.arange(10))
    assert np.all(sigmoid_layer.activation_non_linear == sigmoid(np.arange(10)))


def test_mlp():
    """ testing mlp """

    


test_layer()
test_linear_layer()
test_sigmoid_layer()
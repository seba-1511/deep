"""
Multi-Layer Perceptron
"""

import numpy as np


def sigmoid(x):
    """ element-wise sigmoid activation function """

    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """ element-wise derivative of sigmoid activation function """

    return sigmoid(x) * (1.0 - sigmoid(x))


class Layer():
    """ an abstract layer """


class LinearLayer():
    """ applies a linear transformation to input """


class SigmoidLayer():
    """ applies the sigmoid non-linearity to linear layer """

    def __init__(self, input_size, output_size):
        """ initialize weights uniformly """

        self.input_size = input_size
        self.output_size = output_size
        self.bias = np.zeros((output_size, 1))
        self.weights = np.random.uniform(
            low=-1.0 / input_size,
            high=1.0 / input_size,
            size=((output_size, input_size))
        )

        self.activation_below = None
        self.linear_activation = None
        self.non_linear_activation = None
        self.delta = None

    def fprop(self, activation_below):
        """ forwards propagation """

        self.activation_below = activation_below
        self.linear_activation = np.dot(self.weights, activation_below) + self.bias
        self.non_linear_activation = sigmoid(self.linear_activation)
        return self.non_linear_activation

    def bprop(self, error_above):
        """ backwards propagation """

        self.delta = error_above * sigmoid_prime(self.linear_activation)
        return np.dot(self.weights.T, self.delta)

    def update(self, learn_rate):
        """ update weights """

        self.weights -= learn_rate * np.dot(self.delta, self.activation_below.T)
        self.bias -= learn_rate * self.delta

class ConvolutionalLayer():
    """ applies a convolution to input """


class MaxPoolingLayer():
    """ applies max pooling to a convolutional layer """


class MLP():
    """ a multi-layer perceptron """

    def __init__(self, layers):
        """ initialize mlp with a list of layers """

        assert isinstance(layers, list)
        assert all(isinstance(layer, SigmoidLayer) for layer in layers)
        assert len(layers) >= 1

        self.layers = layers

    def fprop(self, activation):
        """ calculate activation of each layer """

        for layer in self.layers:
             activation = layer.fprop(activation)
        return activation

    def bprop(self, error):
        """ calculate gradient of each layer """

        for layer in self.layers[::-1]:
            error = layer.bprop(error)

    def update(self, learn_rate):
        """ apply gradient update to each layer """

        for layer in self.layers:
            layer.update(learn_rate)

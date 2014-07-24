"""
Multi-Layer Perceptron
"""

import numpy as np


class Layer(object):
    """ abstract layer class """

    def __init__(self):
        """ common initialization """

        self.activation_below = None

    def fprop(self, activation_below):
        """ save weights for bprop """

        self.activation_below = activation_below

    def bprop(self, error):
        """ does not implement bprop """

        raise NotImplementedError

    def update(self, learn_rate):
        """ does not implement update """

        raise NotImplementedError


class LinearLayer(Layer):
    """ applies a linear transformation to input """

    def __init__(self, input_size, output_size):
        """ initialize weights uniformly """

        super(LinearLayer, self).__init__()

        self.delta = None
        self.activation_linear = None

        self.bias = np.zeros(output_size)
        self.weights = np.random.uniform(
            low=-1.0 / input_size,
            high=1.0 / input_size,
            size=((input_size, output_size))
        )

    def fprop(self, activation_below):
        """ forward transformation """

        super(LinearLayer, self).fprop(activation_below)

        self.activation_linear = np.dot(activation_below, self.weights) \
            + self.bias

        return self.activation_linear

    def bprop(self, error):
        """ backward transformation """

        raise NotImplementedError

    def update(self, learn_rate):
        """ update weights """

        raise NotImplementedError


class SigmoidLayer(LinearLayer):
    """ applies the sigmoid non-linearity to linear layer """

    def __init__(self, input_size, output_size):

        super(SigmoidLayer, self).__init__(input_size, output_size)

        self.activation_non_linear = None

    def fprop(self, activation_below):
        """ forwards propagation """

        super(SigmoidLayer, self).fprop(activation_below)

        self.activation_non_linear = self.sigmoid(self.activation_linear)
        return self.activation_non_linear

    def bprop(self, error_above):
        """ backwards propagation """

        self.delta = error_above * self.sigmoid_prime(self.activation_linear)
        return np.dot(self.delta, self.weights.T)

    def update(self, learn_rate):
        """ update weights """

        gradient = np.dot(self.activation_below.T, self.delta / len(self.delta))

        self.weights -= learn_rate * gradient
        self.bias -= learn_rate * self.delta.mean(0)

    def sigmoid(self, activation):
        """ element-wise sigmoid activation function """

        return 1.0 / (1.0 + np.exp(-activation))

    def sigmoid_prime(self, activation):
        """ element-wise derivative of sigmoid activation function """

        return self.sigmoid(activation) * (1.0 - self.sigmoid(activation))


class ConvolutionalLayer():
    """ applies a convolution to input """

    def __init__(self):

        raise NotImplementedError


class MaxPoolingLayer():
    """ applies max pooling to a convolutional layer """

    def __init__(self):

        raise NotImplementedError


class MultiLayerPerceptron(object):
    """ a multi-layer perceptron """

    def __init__(self, layers):
        """ initialize mlp with a list of layers """

        assert isinstance(layers, list)

        # initialize with sigmoid layers sized according to list
        if all(isinstance(layer, int) for layer in layers):

            sigmoid_layers = []

            for in_size, out_size in zip(layers, layers[1:]):
                layer = SigmoidLayer(in_size, out_size)
                sigmoid_layers.append(layer)
            self.layers = sigmoid_layers

        # initialize with layers according to list
        elif all(isinstance(layer, Layer) for layer in layers):

            # TODO: assertion to check if layers match up
            self.layers = layers

        # initialize with bottom layers of autoencoders
        elif all(isinstance(layer, AutoEncoder) for layer in layers):

            raise NotImplementedError

        else:

            raise TypeError('List must be all ints, layers, or autoencoders')

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
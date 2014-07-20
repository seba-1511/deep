"""
Multi-Layer Perceptron
"""

import numpy as np
import load_data


def sigmoid(x):
    """ element-wise sigmoid activation function """

    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """ element-wise derivative of sigmoid activation function """

    return sigmoid(x) * (1.0 - sigmoid(x))


class SigmoidLayer():
    """ a non-linear layer of a multi-layer perceptron """

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

        self.activation_below = activation_below
        self.linear_activation = np.dot(self.weights, activation_below) + self.bias
        self.non_linear_activation = sigmoid(self.linear_activation)
        return self.non_linear_activation

    def bprop(self, error_above):
        """ update weights """

        self.delta = error_above * sigmoid_prime(self.linear_activation)
        return np.dot(self.weights.T, self.delta)

    def update(self):

        self.weights -= 0.1 * np.dot(self.delta, self.activation_below.T)
        self.bias -= 0.1 * self.delta


class MLP():
    """ a multi-layer perceptron """

    def __init__(self, layers):
        """ initialize mlp with a list of layers """

        assert isinstance(layers, list)
        assert all(isinstance(layer, SigmoidLayer) for layer in layers)
        assert len(layers) >= 1

        self.layers = layers

    def fprop(self, input_x):
        """ calculate activation of each layer """

        rval = input_x
        for layer in layers:
             rval = layer.fprop(rval)
        return rval

    def bprop(self, input_x, label_y):
        """ calculate gradient of each layer """

        error = self.fprop(input_x) - label_y
        for layer in self.layers[::-1]:
            error = layer.bprop(error)

    def update(self):
        """ apply gradient update to each layer """

        for layer in layers:
            layer.update()

    def train(self, train_x, train_y):

        # TODO: move train to separate file

        for x, y in zip(train_x, train_y):

            self.bprop(x, y)
            self.update()

        correct = 0.0
        for x, y in zip(train_x, train_y):

            print np.argmax(self.fprop(x)), np.argmax(y)

            if np.argmax(self.fprop(x)) == np.argmax(y):
                correct += 1.0

        return correct / len(train_x)


data = load_data.mnist()
data = load_data.reshape(data)
train_x, train_y = data[0]

layer1 = SigmoidLayer(784, 30)
layer2 = SigmoidLayer(30, 10)
layers = [layer1, layer2]

mlp = MLP(layers)

print mlp.train(train_x, train_y)

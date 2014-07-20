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
    """ a non-linear layer of a mlp """

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

    def linear(self, input_x):
        """ linear output """

        return np.dot(self.weights, input_x) + self.bias

    def non_linear(self, linear):
        """ non-linear output """

        return sigmoid(linear)

    def update(self, input_x, label_y):
        """ update weights """

        linear = self.linear(input_x)
        non_linear = self.non_linear(linear)

        delta = -(label_y - non_linear) * sigmoid_prime(linear)

        self.weights -= 0.1 * np.dot(delta, input_x.T)
        self.bias -= 0.1 * delta

    def score(self, valid_x, valid_y):

        correct = 0.0

        for x, y in zip(valid_x, valid_y):

            linear = self.linear(x)
            non_linear = self.non_linear(linear)

            print non_linear

            if np.argmax(non_linear) == np.argmax(y):
                correct += 1.0

        return correct / len(train_x)

    def predict(self, input_x):

        linear = self.linear(input_x)
        non_linear = self.non_linear(linear)

        return np.argmax(non_linear)

data = load_data.mnist()
data = load_data.reshape(data)
train_x, train_y = data[0]



layer = SigmoidLayer(784, 10)


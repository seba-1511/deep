"""
Multi-Layer Perceptron
"""

import numpy
import load_data

def sigmoid(x):
    """ element-wise sigmoid activation function """

    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_prime(x):
    """ element-wise derivative of sigmoid activation function """

    return sigmoid(x) * (1 - sigmoid(x))


class MLP():

    def __init__(self, input_size, hidden_size, visible_size):
        """ initialize starting weights from uniform distribution """

        self.input_size = input_size
        self.visible_size = visible_size

        self.in_to_hid_weights = numpy.random.uniform(
            low=-1,
            high=1,
            size=(hidden_size, input_size)
        )

        self.hid_to_vis_weights = numpy.random.uniform(
            low=-1,
            high=1,
            size=(visible_size, hidden_size)
        )

        self.in_to_hid_bias = numpy.random.uniform(
            low=-1,
            high=1,
            size=(hidden_size, 1)
        )

        self.hid_to_vis_bias = numpy.random.uniform(
            low=-1,
            high=1,
            size=(visible_size, 1)
        )

    def train(self, train_x, train_y):
        """ train until convergence """

        assert train_x.shape[1] == self.input_size
        assert train_y.shape[1] == self.visible_size
        assert train_x.shape[2] == 1
        assert train_y.shape[2] == 1

        epochs = 10

        for epoch in epochs:

            for x, y in zip(train_x, train_y):

                cost = self.cost(x, y)
                self.update(cost)

                print "epoch", epoch + 1, " cost ", cost

    def valid(self, valid_x, valid_y):
        """ accuracy on labeled data """

        raise NotImplementedError

    def test(self, test_x, test_y):
        """ predictions on unlabeled data """

        raise NotImplementedError

    def activations(self, x):
        """ (hidden linear, non-linear, visible linear, non-linear) """

        activation_list = []

        activation_list.append(numpy.dot(
            self.in_to_hid_weights, x) + self.in_to_hid_bias
        )

        activation_list.append(sigmoid(activation_list[0]))

        activation_list.append(numpy.dot(
            self.hid_to_vis_weights, activation_list[1]) + self.hid_to_vis_bias
        )

        activation_list.append(sigmoid(activation_list[2]))

        return activation_list

    def feed_forward(self, x):
        """ feed forward activation of x """

        hidden_activation = sigmoid(numpy.dot(
            self.in_to_hid_weights, x) +
            self.in_to_hid_bias)

        return sigmoid(numpy.dot(
            self.hid_to_vis_weights, hidden_activation) +
            self.hid_to_vis_bias)

    def cost(self, activation, y):
        """ difference between f(x) and y """

        return activation - y

    def update(self, x, y):
        """ update weights based on x and y """

        activation_list = self.activations(x)

        cost_list = self.cost(activation_list[3], y)

        visible_delta = self.visible_delta(cost_list, activation_list)
        visible_gradient = self.visible_gradient(visible_delta, activation_list)

    def visible_delta(self, cost_list, activation_list):
        """ delta for visible weights """

        return -cost_list * sigmoid_prime(activation_list[2])

    def visible_gradient(self, visible_delta, activation_list):
        """ gradient for visible weights """

        return numpy.dot(visible_delta, activation_list[1].T)

    def hidden_delta(self, cost):
        """ delta for hidden weights """
        raise NotImplementedError

    def hidden_gradient(self, cost, activation_list):
        """ gradient for hidden weights """

        raise NotImplementedError


data = load_data.mnist()
data = load_data.reshape(data)

train_x, train_y = data[0]

mlp = MLP(784, 30, 10)

cost = mlp.update(train_x[0], train_y[0])


"""
Multi-Layer Perceptron
"""

import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d


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
            size=(input_size, output_size))

    def fprop(self, activation_below):
        """ forward transformation """

        super(LinearLayer, self).fprop(activation_below)

        self.activation_linear = np.dot(activation_below, self.weights) \
            + self.bias

        return self.activation_linear

    def bprop(self, error_above):
        """ backward transformation """

        self.delta = error_above
        return np.dot(self.delta, self.weights.T)

    def update(self, learn_rate):
        """ update weights """

        gradient = np.dot(self.activation_below.T, self.delta)
        self.weights -= learn_rate * gradient
        self.bias -= learn_rate * self.delta.mean(0)


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

        error_above *= self.sigmoid_prime(self.activation_linear)
        return super(SigmoidLayer, self).bprop(error_above)

    def update(self, learn_rate):
        """ update weights """

        super(SigmoidLayer, self).update(learn_rate)

    @staticmethod
    def sigmoid(activation):
        """ element-wise sigmoid activation function """

        return 1.0 / (1.0 + np.exp(-activation))

    @staticmethod
    def sigmoid_prime(activation):
        """ element-wise derivative of sigmoid activation function """

        s = SigmoidLayer.sigmoid(activation)

        return s * (1.0 - s)


class LinearConvolutionLayer(Layer):

    def __init__(self, num_filters, filter_size):

        super(LinearConvolutionLayer, self).__init__()

        self.delta = None
        self.activation_linear = None

        self.bias = np.zeros((1, num_filters, 1, 1))
        self.weights = np.random.uniform(
            low=-1.0 / num_filters * filter_size * filter_size,
            high=1.0 / num_filters * filter_size * filter_size,
            size=(num_filters, filter_size, filter_size))

    def fprop(self, activation_below):

        image_shape = np.sqrt(activation_below.shape[-1])
        self.activation_below = activation_below.reshape(-1, image_shape, image_shape)

        activation = []
        for image in self.activation_below:

            feature_map = []
            for weight in self.weights:
                feature_map.append(correlate2d(image, weight, mode='valid'))

            activation.append(feature_map)

        self.activation_linear = np.array(activation) + self.bias
        return self.activation_linear.reshape(-1, self.activation_linear[0].size)

    def bprop(self, error_above):

        self.delta = error_above.reshape(self.activation_linear.shape)

        propagation = []
        for errors in self.delta:

            filter_propagation = []
            for error, weight in zip(errors, self.weights):
                filter_propagation.append(convolve2d(error, weight))
            propagation.append(filter_propagation)

        # sum over filters
        return np.sum(propagation, axis=1)

    def update(self, learn_rate):

        for image, errors in zip(self.activation_below, self.delta):

            for weight, error in zip(self.weights, errors):

                # TODO: add bias update
                weight -= learn_rate * correlate2d(image, error, 'valid')


class SigmoidConvolutionLayer(LinearConvolutionLayer):

    def __init__(self, num_filters, dim_filters):

        super(SigmoidConvolutionLayer, self).__init__(num_filters, dim_filters)
        self.activation_non_linear = None

    def fprop(self, activation_below):

        super(SigmoidConvolutionLayer, self).fprop(activation_below)
        self.activation_non_linear = SigmoidLayer.sigmoid(self.activation_linear)
        return self.activation_non_linear.reshape(-1, self.activation_linear[0].size)

    def bprop(self, error_above):

        error_above = error_above.reshape(self.activation_linear.shape)
        error_above *= SigmoidLayer.sigmoid_prime(self.activation_linear)

        return super(SigmoidConvolutionLayer, self).bprop(error_above)

    def update(self, learn_rate):

        super(SigmoidConvolutionLayer, self).update(learn_rate)


class MaxPoolingLayer(Layer):
    """ applies max pooling to a convolutional layer """

    def __init__(self, pool_size):

        super(MaxPoolingLayer, self).__init__()

        self.pool_size = pool_size

    def fprop(self, activation_below):

        dim = np.array(activation_below).shape[2] / 2

        pools = []

        for image in activation_below:

            activation = []

            for filter in image:

                pool = np.zeros((dim, dim))

                for row in range(0, len(filter)-2, self.pool_size):

                    for col in range(0, len(filter)-2, self.pool_size):

                        pool[row/2][col/2] = np.max(filter[row:row+self.pool_size,
                                                       col:col+self.pool_size])

                pools.append(pool)

            activation.append(pools)

        return np.array(activation).reshape(500, -1)

    def bprop(self, error):

        pass

    def update(self, learn_rate):

        pass


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
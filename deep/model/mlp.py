"""Multi-Layer Perceptron

Layers
------
Linear Layer
Sigmoid Layer
Linear Convolution Layer
Sigmoid Convolution Layer

WIP
---
Max Pooling Layer

Todo
----
Tanh Layer
Mean Pooling Layer
Linear Rectifier Layer
Maxout Layer

References
----------
Notation reflects these tutorials

"Multi-Layer Neural Network"
<http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>

"Convolutional Neural Network"
<http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/>

"Multilayer Perceptron"
<http://deeplearning.net/tutorial/mlp.html#mlp>

"Convolutional Neural Networks (LeNet)"
<http://deeplearning.net/tutorial/lenet.html#lenet>

"""

import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d


class Layer(object):
    """Abstract Layer

    Base class for all layers in this module. Randomly initializes weights
    of all layers

    Parameters
    ----------
    W_shape : tuple
        Shape of weight array

    b_shape : tuples
        Shape of bias array

    Attributes
    ----------
    W : array
        Shape is defined by W_shape

    b : array
        Shape is defined by b_shape

    """
    def __init__(self, W_shape, b_shape):

        self.W_shape = W_shape
        self.b_shape = b_shape

        self.b = np.zeros(b_shape)
        self.W = np.random.uniform(
            low=-1.0 / np.product(W_shape),
            high=1.0 / np.product(W_shape),
            size=W_shape)

    def fprop(self, activation_below):

        raise NotImplementedError

    def bprop(self, error):

        raise NotImplementedError

    def update(self, learn_rate):

        raise NotImplementedError


class LinearLayer(Layer):
    """Linear Layer

    Applies a linear transformation to input. Base for non-linear layers.

    Parameters
    ----------
    n_in : int
        size of input

    n_out : int
        size of output

    Attributes
    ----------
    W : array
        weights

    b : array
        bias

    delta : array
        delta

    z : array
        linear transformation

    """
    def __init__(self, n_in, n_out):
        """ initialize weights uniformly """

        W_shape = n_in, n_out
        b_shape = n_out

        super(LinearLayer, self).__init__(W_shape, b_shape)

        self.delta = None
        self.z = None

    def fprop(self, x):
        """ forward transformation """

        self.x = x
        self.z = np.dot(x, self.W) + self.b

        return self.z

    def bprop(self, error_above):
        """ backward transformation """

        self.delta = error_above
        return np.dot(self.delta, self.W.T)

    def update(self, learn_rate):
        """ update weights """

        gradient = np.dot(self.x.T, self.delta)
        self.W -= learn_rate * gradient
        self.b -= learn_rate * self.delta.mean(0)


class SigmoidLayer(LinearLayer):
    """ applies the sigmoid non-linearity to linear layer """

    def __init__(self, n_in, n_out):

        super(SigmoidLayer, self).__init__(n_in, n_out)

        self.activation_non_linear = None

    def fprop(self, x):
        """ forwards propagation """

        super(SigmoidLayer, self).fprop(x)

        self.activation_non_linear = self.sigmoid(self.z)
        return self.activation_non_linear

    def bprop(self, error_above):
        """ backwards propagation """

        error_above *= self.sigmoid_prime(self.z)
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

        W_shape = num_filters, filter_size, filter_size
        b_shape = 1, num_filters, 1, 1

        super(LinearConvolutionLayer, self).__init__(W_shape, b_shape)

        self.delta = None
        self.activation_linear = None

    def fprop(self, activation_below):

        image_shape = np.sqrt(activation_below.shape[-1])
        self.activation_below = activation_below.reshape(-1, image_shape, image_shape)

        activation = []
        for image in self.activation_below:

            feature_map = []
            for weight in self.W:
                feature_map.append(correlate2d(image, weight, mode='valid'))

            activation.append(feature_map)

        self.activation_linear = np.array(activation) + self.b
        return self.activation_linear.reshape(-1, self.activation_linear[0].size)

    def bprop(self, error_above):

        self.delta = error_above.reshape(self.activation_linear.shape)

        propagation = []
        for errors in self.delta:

            filter_propagation = []
            for error, weight in zip(errors, self.W):
                filter_propagation.append(convolve2d(error, weight))
            propagation.append(filter_propagation)

        # sum over filters
        return np.sum(propagation, axis=1)

    def update(self, learn_rate):

        for image, errors in zip(self.activation_below, self.delta):

            for weight, error in zip(self.W, errors):

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

        print activation_below.shape

    def bprop(self, error):

        pass

    def update(self, learn_rate):

        pass


class MultiLayerPerceptron(Layer):
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


import deep.dataset.mnist
import deep.train.train

M = deep.dataset.mnist.MNIST()
s1 = SigmoidConvolutionLayer(10, 3)
s2 = SigmoidLayer(6760, 10)

mlp = MultiLayerPerceptron([s1, s2])

deep.train.train.bgd(M, mlp)
"""
Feed Forward Neural Network
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
 # License: BSD 3 clause

from deep.layers.convolutional import ConvolutionLayer
from deep.neural_network.base import FeedForwardNN
from deep.layers.base import SigmoidLayer, SoftmaxLayer


n_filters = [20]
filter_sizes = [5]
conv_layer_sizes = zip(n_filters, filter_sizes)
layer_sizes = [100]
layer_types = [SigmoidLayer, SoftmaxLayer]
batch_size = 100
n_iter = 10
learning_rate = 1


class ConvolutionalNN(FeedForwardNN):
    """ """
    def __init__(self, conv_layer_sizes=conv_layer_sizes, layer_sizes=layer_sizes, n_iter=n_iter,
                 batch_size=batch_size, learning_rate=learning_rate, layer_types=layer_types):
        self.conv_layer_sizes = conv_layer_sizes
        self.layer_sizes = layer_sizes
        self.layers = []
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activations = layer_types

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def fit(self, X, y):
        """ """
        _, n_features = X.shape

        for n_filters, filter_size in self.conv_layer_sizes:
            self.layers.append(ConvolutionLayer(n_filters, filter_size, self.batch_size, n_features))

        return super(ConvolutionalNN, self).fit(X, y)

if __name__ == '__main__':
    from deep.datasets import load_mnist
    X, y = load_mnist()[0]
    print ConvolutionalNN().fit(X, y).score(X, y)
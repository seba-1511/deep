"""
Stacked Autoencoder
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

from deep.layers.base import SigmoidLayer
from deep.hyperparams import layer_sizes


class StackedAE(object):

    def __init__(self, layer_sizes=layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    def transform(self, X):
        for autoencoder in self.layers:
            X = autoencoder.transform(X)
        return X

    def inverse_transform(self, X):
        for autoencoder in self.layers[::-1]:
            X = autoencoder.inverse_transfom
        return X

    def fit(self, X):
        n_features = X.shape[1]

        for shape in zip([n_features] + self.layer_sizes, self.layer_sizes):
            self.layers.append(SigmoidLayer(shape))

        for autoencoder in self.layers:
            X = autoencoder.fit_transform(X)
        return self

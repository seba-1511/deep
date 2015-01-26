import matplotlib.pyplot as plt

from deep.corruptions import SaltAndPepper

import numpy as np


def setup_plot(func):

    def wrapper(X, filename=None, **kwargs):
        n_samples = X.shape[0]
        dim = int(np.sqrt(n_samples))
        plt.figure(figsize=(dim, dim))
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0,
                    top=1.0, wspace=0.0, hspace=0.0)

        func(X, **kwargs)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    return wrapper


@setup_plot
def plot_data(X, filename=None):

    for index, x in enumerate(X):
        plt.setp(plt.subplot(10, 10, index), xticks=[], yticks=[])
        plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray'),
                   interpolation='nearest')


@setup_plot
def plot_corruption(X, corrupt=SaltAndPepper, filename=None):

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0,
                        top=1.0, wspace=0.0, hspace=0.0)

    corruptors = [corrupt(level/10.) for level in range(1, 10)]

    for row, x in enumerate(X):

        corrupted = [corrupt(x) for corrupt in corruptors]
        corrupted = [x] + corrupted

        for col, c in enumerate(corrupted):
            plt.setp(plt.subplot(10, 10, row*10+col+1), xticks=[], yticks=[])
            plt.imshow(c.reshape(28, 28), cmap=plt.get_cmap('gray'),
                       interpolation='nearest')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_layer(layer, filename=None):
    from deep.layers import ConvolutionLayer

    if isinstance(layer, ConvolutionLayer):
        raise NotImplemented

    plot_data(layer.W.get_value().T)


@setup_plot
def plot_reconstruction(autoencoder):
    #: check if denoising autoencoder.
    raise NotImplementedError


@setup_plot
def plot_classification_errors(classifier):
    raise NotImplementedError


def plot_fit(fit):
    raise NotImplementedError
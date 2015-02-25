import matplotlib.pyplot as plt

from deep.corruptions import SaltAndPepper
from deep.plot.helper import plotLines

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
def plot_image_data(X, filename=None):

    for index, x in enumerate(X):
        plt.setp(plt.subplot(10, 10, index + 1), xticks=[], yticks=[])
        plt.imshow(x.reshape(28, 28), cmap=plt.get_cmap('gray'),
                   interpolation='nearest')


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


def plot_training(model, name='training'):
    train = model.fit_method.train_scores
    valid = model.fit_method.valid_scores
    index = [i for i, _ in enumerate(valid)]
    train = (index, train)
    valid = (index, valid)
    plotLines((train, valid), title=name, xlabel='Iterations',
              ylabel='Score', yscale='log')

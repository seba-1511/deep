""" Tied Wieght Autoencoder
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import time
import theano.tensor as T
import numpy as np

from abc import ABCMeta, abstractmethod

from theano import function
from theano import shared
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin


def _salt_pepper(x, p=0.5, rng=None):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.

        """
        if not rng:
            rng = RandomStreams(0)
        a = rng.binomial(size=x.shape, p=1-p, dtype='float32')
        b = rng.binomial(size=x.shape, p=0.5, dtype='float32')
        c = T.eq(a, 0) * b
        return x * a + c


def _gaussian(x, std=0.5, rng=None):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.

        """
        if not rng:
            rng = RandomStreams(0)
        return x + rng.normal(size=x.shape, std=std, dtype=theano.config.floatX)


_corruptions = {'salt_pepper': _salt_pepper, 'gaussian': _gaussian, }
_activations = {'sigmoid': sigmoid, 'tanh': tanh}


class BaseAE(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """Tied Weight Autoencoder (AE).

    Description.

    Parameters
    ----------
    n_hiddens : int, optional
        Number of hidden units.

    learning_rate : float, optional
        The learning rate for weight updates.

    batch_size : int, optional
        Number of examples per mini-batch.

    n_iter : int, optional
        Number of iterations over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    Attributes
    ----------
    b_encode : array-like, shape (n_hiddens,)
        Biases of the hidden units.

    b_decode : array-like, shape (n_features,)
        Biases of the visible units.

    W_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_hiddens is the number of hidden units.

    Examples
    --------
    >>> import numpy as np
    >>> from deep.autoencoder import BaseAE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BaseAE()
    >>> model.fit(X)
    TiedWeightAE(batch_size=10, learning_rate=1, n_hiddens=10, n_iter=10,
           verbose=0)

    References
    ----------
    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """


    @abstractmethod
    def __init__(self, n_hidden, activation, tied, corruption,
                 learning_rate, batch_size, n_iter, rng, verbose):

        if activation in _activations:
            self.activation = _activations[activation]
        else:
            raise ValueError('Activation should be one of %s, %s was given'
                             % _activations.keys(), activation)

        if corruption and corruption not in _corruptions:
            raise ValueError('Corruption should be one of %s, %s was given'
                 % (_corruptions.keys(), corruption))

        self.corruption = corruption
        self.n_hidden = n_hidden
        self.tied = tied
        self.corruption = corruption
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.rng = RandomStreams(0)
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        n_samples, n_features = X.shape

        self.b_encode_ = shared(np.zeros(self.n_hidden, dtype='float32'))
        self.b_decode_ = shared(np.zeros(n_features, dtype='float32'))

        self.W_encode_= shared(np.asarray(np.random.uniform(
            low=-np.sqrt(6. / (n_features + self.n_hidden)),
            high=np.sqrt(6. / (n_features + self.n_hidden)),
            size=(n_features, self.n_hidden)), dtype='float32'))

        if self.tied:
            self.W_decode_ = self.W_encode_.T
            params = [self.W_encode_, self.b_encode_, self.b_decode_]
        else:
            self.W_decode_= shared(np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (n_features + self.n_hidden)),
                high=np.sqrt(6. / (n_features + self.n_hidden)),
                size=(n_features, self.n_hidden)), dtype='float32'))
            params = [self.W_encode_, self.b_encode_,
                      self.W_decode_, self.b_decode_]

        if self.corruption:
            x = _corruptions[self.corruption]((T.fmatrix()))
        else:
            x = T.fmatrix()
        encode = self.activation(T.dot(x, self.W_encode_) + self.b_encode_)
        decode = self.activation(T.dot(encode, self.W_decode_) + self.b_decode_)
        score = T.mean(T.nnet.binary_crossentropy(decode, x))

        gradients = T.grad(score, params)
        updates = [(param, param - self.learning_rate * grad)
                   for param, grad in zip(params, gradients)]

        X = shared(np.asarray(X, dtype='float32'))
        index = T.lscalar()
        indexed_batch = X[index*self.batch_size:(index+1)*self.batch_size]
        givens = {x:indexed_batch}
        fit_function = function([index], score, updates=updates, givens=givens)

        begin = time.time()
        n_batches = n_samples / self.batch_size
        self.scores_ = []
        for iteration in range(1, self.n_iter + 1):
            cost = [fit_function(batch_index)
                    for batch_index in range(n_batches)]
            self.scores_.append(np.mean(cost))

            if self.verbose:
                end = time.time()
                print("[%s] Iteration %d, cost = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                        self.scores_[-1], end - begin))
                begin = end

        return self

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        X is encoded by an affine transformation followed by a non-linearity.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_hiddens)

        """
        return self.activation(T.dot(X, self.W_encode_) + self.b_encode_)

    def inverse_transform(self, X):
        """Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        """
        return self.activation(T.dot(X, self.W_encode_.T) + self.b_decode_)

    def score(self, X):
        """Compute the reconstruction error of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        error : float
            The average reconstruction error of X.

        """
        encoded = self.activation(T.dot(X, self.W_encode_) + self.b_encode_)
        decoded = self.activation(T.dot(encoded, self.W_encode_.T) + self.b_decode_)
        return T.mean(T.nnet.binary_crossentropy(decoded, X)).eval()





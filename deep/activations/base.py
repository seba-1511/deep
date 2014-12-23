"""
Activations
"""

# Author: Gabriel Pereyra <gbrl.pereyra@gmail.com>
#
# License: BSD 3 clause

import theano.tensor as T
from abc import abstractproperty


class Activation(object):

    activation = abstractproperty


class Linear(Activation):

    activation = lambda x: x


class Sigmoid(Activation):

    activation = T.nnet.sigmoid


class Softmax(Activation):

    activation = T.nnet.softmax


class Tanh(Activation):

    activation = T.tanh


class RectifiedLinear(Activation):

    pass


class Maxout(Activation):

    pass
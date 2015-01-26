# -*- coding: utf-8 -*-
"""
    deep.updates.base
    -----------------

    Implements various types of update methods.

    :references: pylearn2 (optimization module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import theano
import theano.tensor as T

from abc import abstractmethod


class Update(object):
    """An abstract class that represents an update method, typically
    in a denoising autoencoder (DAE). Once initialized, an update class
    transforms inputs through its __call__ method.

    Example::

        update = GradientDescent()
        updated_input = update(input)

    """
    @abstractmethod
    def __call__(self, cost, param, learning_rate):
        """"""

    def __repr__(self):
        return str(self.__class__.__name__)


class GradientDescent(Update):
    """Gradient descent updates a param by the gradient of a cost function,
    scaled by a learning rate

    :reference: ???

    :param x: a tensor_like Theano symbolic representing the input.
    :return: a corrupted Theano symbolic of same dims as the input.
    """
    def __call__(self, cost, param, learning_rate):
        return [(param, param - learning_rate * T.grad(cost, param))]


class Momentum(Update):
    """Gradient descent with momentum adds a velocity update in addition to
    the standard gradient descent update.

    :reference: ???

    :param momentum: a tensor_like Theano symbolic representing the input.
    """
    def __init__(self, momentum=.5):
        self.momentum = momentum

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        scaled_lr = learning_rate * lr_scalers.get(param, 1.)

        grad = T.grad(cost, param)
        vel = theano.shared(param.get_value() * 0.)

        inc = self.momentum * vel - scaled_lr * grad

        return [(param, param + inc), (vel, inc)]


class NesterovMomentum(Momentum):
    """Gradient descent with Nesterov momentum adds a velocity update in addition to
    the standard gradient descent update.

    :reference: ???

    :param momentum: a tensor_like Theano symbolic representing the input.
    """
    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        scaled_lr = learning_rate * lr_scalers.get(param, 1.)

        grad = T.grad(cost, param)
        vel = theano.shared(param.get_value() * 0.)

        inc = self.momentum * vel - scaled_lr * grad
        inc = self.momentum * inc - scaled_lr * grad

        return [(param, param + inc), (vel, inc)]


class AdaDelta(Update):
    """Gradient descent with Nesterov momentum adds a velocity update in addition to
    the standard gradient descent update.

    :reference: ???

    :param momentum: a tensor_like Theano symbolic representing the input.
    """
    def __init__(self, decay=0.95):
        self.decay=decay

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        grad = T.grad(cost, param)

        mean_square_grad = theano.shared(param.get_value() * .0)
        mean_square_dx = theano.shared(param.get_value() * 0.)

        new_mean_squared_grad = (
            self.decay * mean_square_grad +
            (1 - self.decay) * T.sqr(grad)
        )

        epsilon = lr_scalers.get(param, 1.) * learning_rate
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

        new_mean_square_dx = (
            self.decay * mean_square_dx +
            (1 - self.decay) * T.sqr(delta_x_t)
        )

        return [(mean_square_grad, new_mean_squared_grad),
                (mean_square_dx, new_mean_square_dx),
                (param, param + delta_x_t)]


class RMSProp(Update):
    """Gradient descent with Nesterov momentum adds a velocity update in addition to
    the standard gradient descent update.

    :reference: ???

    :param momentum: a tensor_like Theano symbolic representing the input.
    """
    def __init__(self, decay=0.1, max_scaling=1e5):
        self.decay = decay
        self.max_scaling = max_scaling
        self.epsilon = 1. / max_scaling

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        grad = T.grad(cost, param)

        mean_square_grad = theano.shared(param.get_value() * 0.)

        new_mean_squared_grad = (
            self.decay * mean_square_grad +
            (1 - self.decay) * T.sqr(grad)
        )

        scaled_lr = lr_scalers.get(param, 1.) * learning_rate
        rms_grad_t = T.sqrt(new_mean_squared_grad)
        rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
        delta_x_t = -scaled_lr * grad / rms_grad_t

        return [(mean_square_grad, new_mean_squared_grad),
                (param, param + delta_x_t)]


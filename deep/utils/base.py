# -*- coding: utf-8 -*-
"""
    deep.utils.base
    ---------------

    Implements various utility functions.

    :references: theano deep learning tutorial

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import theano.tensor as T

from collections import OrderedDict, defaultdict


def numpy_to_theano(x):
    """Maps numpy array to the corresponding TensorVariable type in theano.
    This is used in the theano decorator to allow for arbitrary arguments.

    :param x:
    :return:
    """

    #: is their a theano library func that does this? (as_tensor_type is close)

    numpy_to_theano_dict = defaultdict(dict)

    numpy_to_theano_dict[0]['int32'] = T.iscalar()
    numpy_to_theano_dict[1]['int32'] = T.ivector()
    numpy_to_theano_dict[2]['int32'] = T.imatrix()
    numpy_to_theano_dict[3]['int32'] = T.itensor3()
    numpy_to_theano_dict[4]['int32'] = T.itensor4()

    numpy_to_theano_dict[0]['int64'] = T.lscalar()
    numpy_to_theano_dict[1]['int64'] = T.lvector()
    numpy_to_theano_dict[2]['int64'] = T.lmatrix()
    numpy_to_theano_dict[3]['int64'] = T.ltensor3()
    numpy_to_theano_dict[4]['int64'] = T.ltensor4()

    numpy_to_theano_dict[0]['float32'] = T.fscalar()
    numpy_to_theano_dict[1]['float32'] = T.fvector()
    numpy_to_theano_dict[2]['float32'] = T.fmatrix()
    numpy_to_theano_dict[3]['float32'] = T.ftensor3()
    numpy_to_theano_dict[4]['float32'] = T.ftensor4()

    numpy_to_theano_dict[0]['float64'] = T.dscalar()
    numpy_to_theano_dict[1]['float64'] = T.dvector()
    numpy_to_theano_dict[2]['float64'] = T.dmatrix()
    numpy_to_theano_dict[3]['float64'] = T.dtensor3()
    numpy_to_theano_dict[4]['float64'] = T.dtensor4()

    x = np.asarray(x)
    ndim = x.ndim
    dtype = str(x.dtype)
    return numpy_to_theano_dict[ndim][dtype]


def theano_compatible(func):
    """A decorator that wraps a function which represents a theano expression.
    If the input is a theano tensor variable, then the decorator simply returns
    the expression. Otherwise, the decorator maps the arguments to their
    corresponding theano types and evaluates the theano expression using the
    function arguments as arguments to the theano expression

    :param func:
    """

    #: needs a better name

    def theano_compatible(self, *args):

        are_theano_vars = [isinstance(arg, T.TensorVariable) for arg in args]

        if all(are_theano_vars):
            return func(self, *args)
        else:
            arg_dict = OrderedDict((numpy_to_theano(arg), arg) for arg in args)
            return func(self, *arg_dict.keys()).eval(arg_dict)

    #: renamed 'wrapper' to 'theano_compatible'
    #: we can use this to check if a corruption is theano compatible or not
    #:
    #: in order for this to work we need to remove @wraps.
    #: to compensate, we append the original function name to the new name
    #: and copy the original doc.
    #:
    #: what is a cleaner way to do this?
    try:
        theano_compatible.__name__ = func.__name__
        theano_compatible.__doc__ = func.__doc__
    except AttributeError:
        #: most activations are theano lib functions which don't have a
        #: __name__ attribute (check this)
        pass

    return theano_compatible
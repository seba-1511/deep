from base import BaseAE


class TiedAE(BaseAE):
    """This function does something.

    Args:
       name (str):  The name to use.

    Kwargs:
       state (bool): Current state to be in.

    Returns:
       int.  The return code::

          0 -- Success!
          1 -- No good.
          2 -- Try again.

    Raises:
       AttributeError, KeyError

    A really great idea.  A way you might use me is

    >>> print public_fn_with_googley_docstring(name='foo', state=None)
    0

    BTW, this always returns 0.  **NEVER** use with :class:`MyPublicClass`.

    """
    def __init__(self, n_hidden=10, activation='sigmoid', tied=True,
                 corruption=None, learning_rate=1, batch_size=10,
                 n_iter=10, rng=None, verbose=0):
        super(TiedAE, self).__init__(n_hidden, activation, tied,
                                     corruption, learning_rate, batch_size, \
                                     n_iter, rng, verbose)


class UntiedAE(BaseAE):

    def __init__(self, n_hidden=10, activation='sigmoid', tied=False,
                 corruption=None, learning_rate=1, batch_size=10,
                 n_iter=10, rng=None, verbose=0):
        super(UntiedAE, self).__init__(n_hidden, activation, tied,
                                       corruption, learning_rate, batch_size,
                                       n_iter, rng, verbose)


class DenoisingAE(BaseAE):

    def __init__(self, n_hidden=10, activation='sigmoid', tied=True,
                 corruption='salt_pepper', learning_rate=1, batch_size=10,
                 n_iter=10, rng=None, verbose=0):
        super(DenoisingAE, self).__init__(n_hidden, activation, tied,
                                          corruption, learning_rate, batch_size,
                                          n_iter, rng, verbose)


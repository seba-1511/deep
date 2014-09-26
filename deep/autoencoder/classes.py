from deep.autoencoder.base import BaseAE


class TiedAE(BaseAE):

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


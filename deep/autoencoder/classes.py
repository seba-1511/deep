from deep.autoencoder.base import AutoencoderBase


class TiedAutoencoder(AutoencoderBase):

    def __init__(self, encoder, learning_rate=1,
                 batch_size=10, n_iter=10, verbose=1):
        super(TiedAutoencoder, self).__init__(encoder, learning_rate,
                                              batch_size, n_iter,
                                              verbose)


class TiedDenoisingAutoencoder(AutoencoderBase):

    def __init__(self, encoder, corruption, learning_rate=1,
                 batch_size=10, n_iter=10, verbose=1):
        super(TiedDenoisingAutoencoder, self).__init__(encoder, learning_rate,
                                                       batch_size, n_iter,
                                                       verbose)
        self.corruption = corruption


class UntiedAutoencoder(AutoencoderBase):

    def __init__(self, encoder, decoder, learning_rate=1,
                 batch_size=10, n_iter=10, verbose=1):
        super(UntiedAutoencoder, self).__init__(encoder, learning_rate,
                                                batch_size, n_iter,
                                                verbose)
        self.decoder = decoder

    def fit(self, X, y=None):
        self.decoder._init_params((X.shape[1], self.decoder.layer_size))
        super(UntiedAutoencoder, self).fit(X, y)


class UntiedDenoisingAutoencoder(AutoencoderBase):

    def __init__(self, encoder, decoder, corruption, learning_rate=1,
                 batch_size=10, n_iter=10, verbose=1):
        super(UntiedDenoisingAutoencoder, self).__init__(encoder, learning_rate,
                                                         batch_size, n_iter,
                                                         verbose)
        self.decoder = decoder
        self.corruption = corruption
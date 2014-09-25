from deep.autoencoder.base import AutoencoderBase
from deep.layer import SigmoidLayer
from deep.corruption import SaltPepperCorruption

ENCODER = SigmoidLayer(10)
DECODER = SigmoidLayer(10)
CORRUPTION = SaltPepperCorruption(.25)


class TiedAutoencoder(AutoencoderBase):

    def __init__(self, encoder=ENCODER,
                 learning_rate=1, batch_size=10, n_iter=10, verbose=1):
        super(TiedAutoencoder, self).__init__(encoder, learning_rate,
                                              batch_size, n_iter,
                                              verbose)


class TiedDenoisingAutoencoder(AutoencoderBase):

    def __init__(self, encoder=ENCODER, corruption=CORRUPTION,
                 learning_rate=1, batch_size=10, n_iter=10, verbose=1):
        super(TiedDenoisingAutoencoder, self).__init__(encoder, learning_rate,
                                                       batch_size, n_iter,
                                                       verbose)
        self.corruption = corruption


class UntiedAutoencoder(AutoencoderBase):

    def __init__(self, encoder=ENCODER, decoder=DECODER,
                 learning_rate=1, batch_size=10, n_iter=10, verbose=1):
        super(UntiedAutoencoder, self).__init__(encoder, learning_rate,
                                                batch_size, n_iter,
                                                verbose)
        self.decoder = decoder

    def fit(self, X, y=None):
        self.decoder._init_params(X.shape[1])
        super(UntiedAutoencoder, self).fit(X, y)


class UntiedDenoisingAutoencoder(AutoencoderBase):

    def __init__(self, encoder=ENCODER, decoder=DECODER, corruption=CORRUPTION,
                 learning_rate=1, batch_size=10, n_iter=10, verbose=1):
        super(UntiedDenoisingAutoencoder, self).__init__(encoder, learning_rate,
                                                         batch_size, n_iter,
                                                         verbose)
        self.decoder = decoder
        self.corruption = corruption


class TiedNoisyAutoencoder(AutoencoderBase):

    def __init__(self):
        raise NotImplemented


class UnTiedNoisyAutoencoder(AutoencoderBase):

    def __init__(self):
        raise NotImplemented


class NearestNeighborAutoEncoder(AutoencoderBase):

    def __init__(self, encoder=SigmoidLayer(400, .3),
                 learning_rate=1, batch_size=20, n_iter=20, verbose=1):
        super(NearestNeighborAutoEncoder, self).__init__(encoder, learning_rate,
                                                         batch_size, n_iter,
                                                         verbose)


def plot(images, n_rows, n_cols):
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    for index, image in enumerate(images):
        plt.setp(plt.subplot(n_rows, n_cols, index+1), xticks=[], yticks=[])
        plt.imshow(image, cmap=matplotlib.cm.gray, interpolation="nearest")
    plt.show()

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    X = load_digits().data

    from deep.datasets import load_mnist


    X -= X.min()
    X /= X.max()
    X = X > .5

    X = load_mnist()[0][0]


    nnae = NearestNeighborAutoEncoder()
    nnae.fit(X)

    plot(nnae.encoder.W.get_value().reshape(400, 28, 28), 20, 20)
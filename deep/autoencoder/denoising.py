from deep.autoencoder.base import TiedAE
from deep.layers.denoising import DenoisingLayer
from deep.corruptions.base import SaltAndPepper

corruption = SaltAndPepper()


class TiedDenoisingAE(TiedAE, DenoisingLayer):

    def __init__(self, corruption=corruption):
        super(TiedDenoisingAE, self).__init__()
        self.corrupt = corruption.corrupt


if __name__ == '__main__':
    tdae = TiedDenoisingAE()
    from deep.datasets import load_mnist
    X, y = load_mnist()[0]

    tdae.fit(X)
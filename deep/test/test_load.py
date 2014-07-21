from deep.dataset.mnist import MNIST
from deep.dataset.kaggle_decoding import KaggleDecoding


def test_mnist():
    """ testing mnist initialization """

    MNIST()


def test_kaggle_decoding():
    """ testing kaggle decoding initialization """

    KaggleDecoding()
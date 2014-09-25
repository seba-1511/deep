from sklearn.datasets import load_diabetes
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_array_equal

from numpy import diff

from deep.autoencoder import TiedAutoencoder
from deep.autoencoder import TiedDenoisingAutoencoder
from deep.autoencoder import UntiedAutoencoder
from deep.autoencoder import UntiedDenoisingAutoencoder

from deep.layer import SigmoidLayer
from deep.layer import TanhLayer


def test_default_init_fit():
    """
    Test autoencoder fit methods.
    """
    diabetes = load_diabetes()
    for clf in (TiedAutoencoder(),
                TiedDenoisingAutoencoder(),
                UntiedAutoencoder(),
                UntiedDenoisingAutoencoder()):

        print clf

        clf.fit(diabetes.data)
        assert_less(clf.score(diabetes.data), 0.06)
        assert_array_equal(diff(clf.scores_) < 0, True)


def test_grid_search_fit():
    from sklearn import grid_search

    pass


if __name__ == '__main__':
    import nose
    nose.runmodule()

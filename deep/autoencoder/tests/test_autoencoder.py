from sklearn.datasets import load_diabetes
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_array_equal

from numpy import diff

from deep.autoencoder import TiedAE
from deep.autoencoder import UntiedAE
from deep.autoencoder import DenoisingAE


def test_fit():
    """
    Test autoencoder fit methods.
    """
    diabetes = load_diabetes()
    for clf in (TiedAE(),
                UntiedAE(),
                DenoisingAE()):
        clf.fit(diabetes.data)
        assert_less(clf.score(diabetes.data), 1)
        assert_array_equal(diff(clf.scores_) < 0, True)


if __name__ == '__main__':
    import nose
    nose.runmodule()

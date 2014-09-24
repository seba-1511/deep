from sklearn.datasets import load_diabetes
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_array_equal

from numpy import diff

from deep.autoencoder import TiedAutoencoder
from deep.autoencoder import TiedDenoisingAutoencoder
from deep.autoencoder import UntiedAutoencoder
from deep.autoencoder import UntiedDenoisingAutoencoder

from deep.layer import SigmoidLayer

def test_fit():
    """
    Test autoencoder fit methods.
    """
    diabetes = load_diabetes()
    encoder = SigmoidLayer(10)
    decoder = SigmoidLayer(10)
    corruption = 0.25
    for clf in (TiedAutoencoder(encoder, .1),
                TiedDenoisingAutoencoder(encoder, corruption, .1),
                UntiedAutoencoder(encoder, decoder, .1),
                UntiedDenoisingAutoencoder(encoder, decoder, corruption, .1)):
        clf.fit(diabetes.data)
        assert_less(clf.score(diabetes.data), 0.06)
        assert_array_equal(diff(clf.scores_) < 0, True)

"""
def test_grid_search_fit():
    from sklearn import grid_search

    diabetes = load_diabetes()

    parameters = {'encoder':('linear', 'rbf'), 'C':[1, 10]}
    ae = TiedAutoencoder()
    clf = grid_search.GridSearchCV(ae, parameters)
    clf.fit(diabetes.data)
"""

if __name__ == '__main__':
    import nose
    nose.runmodule()

from sklearn.utils.testing import assert_equal

from deep.layer import SigmoidLayer
from deep.layer import TanhLayer

from numpy.random import random



def test_default_init_encode():
    x = random(100)
    for layer in (SigmoidLayer(),
                  TanhLayer()):
        layer._init_params(100)
        assert_equal(layer.encode(x).shape, (10,))


def test_defualt_init_decode():
    x = random(10)
    for layer in (SigmoidLayer(),
                  TanhLayer()):
        layer._init_params(100)

        # need decode layer to map back to input size
        # decode layer shouldn't take a size param

        print layer.decode(x).shape

        assert_equal(layer.decode(x).shape, (100,))


if __name__ == '__main__':
    import nose
    nose.runmodule()

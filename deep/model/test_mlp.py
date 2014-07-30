import numpy as np
import mlp


class TestLayer():

    def __init__(self):

        self.layer = mlp.Layer()

    def test_fprop(self):

        self.layer.fprop(np.arange(10))

        assert np.all(self.layer.activation_below == np.arange(10))


class TestLinearLayer():

    def __init__(self):

        self.layer = mlp.LinearLayer(10, 5)

    def test_init(self):

        assert self.layer.weights.shape == (10, 5)

    def test_fprop(self):

        self.layer.weights = np.eye(10, 5)
        assert np.all(self.layer.fprop(np.arange(10)) == np.arange(5))

    def test_bprop(self):

        assert False # Todo: implement linear bprop

    def test_update(self):

        assert False # Todo: implement linear update


class TestSigmoidLayer():

    def __init__(self):

        self.layer = mlp.SigmoidLayer(10, 10)

    def test_fprop(self):

        self.layer.weights = np.eye(10, 10)
        assert np.all(self.layer.fprop(np.arange(10)) ==
                      mlp.SigmoidLayer.sigmoid(np.arange(10)))

    def test_bprop(self):

        self.layer.weights = np.eye(10, 10)
        self.layer.activation_linear = np.arange(10)
        assert np.all(self.layer.bprop(np.ones(10)) ==
                      mlp.SigmoidLayer.sigmoid_prime(np.arange(10)))

    def test_update(self):

        self.layer.weights = np.eye(10, 10)
        self.layer.activation_below = np.ones(10)
        self.layer.delta = np.ones(10)
        self.layer.update(1)

        assert np.all(self.layer.weights == np.eye(10, 10) - 10)

"""
class TestLinearConvolutionLayer():

    def __init__(self):

        raise NotImplementedError

    def test_fprop(self):

        raise NotImplementedError

    def test_bprop(self):

        raise NotImplementedError

    def test_update(self):

        raise NotImplementedError


class TestSigmoidConvolutionLayer():

    def __init__(self):

        raise NotImplementedError

    def test_fprop(self):

        raise NotImplementedError

    def test_bprop(self):

        raise NotImplementedError

    def test_update(self):

        raise NotImplementedError


class TestMultiLayerPerceptron():

    def __init__(self):

        raise NotImplementedError

    def test_fprop(self):

        raise NotImplementedError

    def test_bprop(self):

        raise NotImplementedError

    def test_update(self):

        raise NotImplementedError
"""
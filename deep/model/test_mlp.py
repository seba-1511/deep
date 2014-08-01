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

        self.layer = mlp.LinearLayer(10, 10)

    def test_convergence(self):

        for i in range(10):

            error = self.layer.fprop(np.ones((1, 10))) - np.ones(10)
            self.layer.bprop(error)
            self.layer.update(.1)

        assert np.allclose(self.layer.fprop(np.ones((1, 10))), np.ones(10))


class TestSigmoidLayer():

    def __init__(self):

        self.layer = mlp.SigmoidLayer(10, 10)

    def test_convergence(self):

        for i in range(10):

            error = self.layer.fprop(np.ones((1, 10))) - np.ones(10)
            self.layer.bprop(error)
            self.layer.update(100)

        assert np.allclose(self.layer.fprop(np.ones((1, 10))), np.ones(10))


class TestLinearConvolutionLayer():

    def __init__(self):

        self.layer = mlp.LinearConvolutionLayer(5, 2)

    def test_convergence(self):

        for i in range(10):

            error = self.layer.fprop(np.ones((1, 9))) - np.ones(20)

            self.layer.bprop(error)
            self.layer.update(.10)
            print np.sum(error**2)

        print self.layer.fprop(np.ones((1, 9)))

        assert np.allclose(self.layer.fprop(np.ones((1, 10))), np.ones(10))


"""
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
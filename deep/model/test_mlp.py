import numpy as np
import mlp


class TestLinearLayer():

    def __init__(self):

        self.layer = mlp.LinearLayer(10, 2)

    def test_convergence(self):
        """ testing linear layer convergence """

        x = np.arange(10).reshape(1, 10)
        y = np.arange(2).reshape(1, 2)

        for i in range(100):

            error = self.layer.fprop(x) - y
            self.layer.bprop(error)
            self.layer.update(.001)

        assert np.allclose(self.layer.fprop(x), y)


class TestSigmoidLayer():

    def __init__(self):

        self.layer = mlp.SigmoidLayer(10, 2)

    def test_convergence(self):
        """ testing sigmoid layer convergence """

        x = np.arange(10).reshape(1, 10)
        y = np.arange(2).reshape(1, 2)

        for i in range(100):

            error = self.layer.fprop(x) - y
            self.layer.bprop(error)
            self.layer.update(10)

        assert np.allclose(self.layer.fprop(x), y)


class TestLinearConvolutionLayer():

    def __init__(self):

        self.layer = mlp.LinearConvolutionLayer(2, 2)

    def test_convergence(self):
        """ testing linear convolution layer convergence """

        x = np.arange(9).reshape(1, 9)
        y = np.ones(8).reshape(1, 8)

        for i in range(100):

            error = self.layer.fprop(x) - y
            self.layer.bprop(error)
            self.layer.update(.005)

        assert np.allclose(self.layer.fprop(x), y, atol=0.1, rtol=1)


class TestSigmoidConvolutionLayer():

    def __init__(self):

        self.layer = mlp.SigmoidConvolutionLayer(2, 2)

    def test_convergence(self):
        """ testing sigmoid convolution layer convergence """

        x = np.arange(9).reshape(1, 9)
        y = np.ones(8).reshape(1, 8)

        for i in range(100):

            error = self.layer.fprop(x) - y
            self.layer.bprop(error)
            self.layer.update(10)

        assert np.allclose(self.layer.fprop(x), y, atol=0.1, rtol=1)


class TestMultiLayerPerceptron():

    def __init__(self):

        layer1 = mlp.LinearLayer(10, 5)
        layer2 = mlp.LinearLayer(5, 2)
        self.mlp = mlp.MultiLayerPerceptron([layer1, layer2])

    def test_convergence(self):
        """ testing multi-layer perceptron convergence """

        x = np.arange(10).reshape(1, 10)
        y = np.arange(2).reshape(1, 2)

        for i in range(100):

            error = self.mlp.fprop(x) - y
            self.mlp.bprop(error)
            self.mlp.update(.01)

        assert np.allclose(self.mlp.fprop(x), y)


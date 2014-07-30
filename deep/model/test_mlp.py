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


class TestLinearConvolutionLayer():

    def __init__(self):

        self.layer = mlp.LinearConvolutionLayer(5, 2)

    def test_init(self):

        assert self.layer.weights.shape == (5, 2, 2)

    def test_fprop(self):

        # set each filter to 2x2 identity matrix
        for i in range(5):
            self.layer.weights[i] = np.eye(2)

        # set input to 3x3 identity matrix and flatten
        image = np.eye(3).reshape(9)

        # compare input to weights * 2
        assert np.all(self.layer.fprop(image).reshape(5, 2, 2)
                      == self.layer.weights * 2)

    def test_bprop(self):

        # set weights and error to ones
        self.layer.weights = np.ones((5, 2, 2))
        error = np.ones((5, 2, 2))

        # convolved size is image size - filter size + 1 = 3 - 2 + 1
        self.layer.convolved_image_size = 2




        print self.layer.bprop(error).shape

    def test_update(self):

        raise NotImplementedError


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
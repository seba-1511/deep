"""
MNIST Data Class
"""

import gzip
import cPickle
from dataset import DataSet


class MNIST(DataSet):
    """ MNIST dataset from <link> """

    def __init__(self):

        print "loading mnist digits dataset ... ",
        data = self.load()
        print "ok"

        self.train_set = self.reshape(data[0])
        self.valid_set = self.reshape(data[1])
        self.test_set = self.reshape(data[2])

        print
        print "x dimensions ...", self.train_set[0][0].shape
        print "y dimensions ...", self.train_set[1][0].shape

    def load(self):

        # TODO: pass file name to super to load?

        f = gzip.open("data/mnist.pkl.gz")
        data = cPickle.load(f)
        f.close()

        return data

    def plot_classes(self):
        """ plot mnist digits 1-9 """

        # TODO: plot mnist digits 1-9

        raise NotImplementedError

m = MNIST()
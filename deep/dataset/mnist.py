"""
MNIST Data Class
"""

import gzip
import cPickle
from dataset import DataSet


class MNIST(DataSet):
    """ MNIST dataset from <link> """

    @wraps
    def load(self):

        # TODO: pass file name to super to load?

        f = gzip.open("data/mnist.pkl.gz")
        data = cPickle.load(f)
        f.close()

        return data


m = MNIST()
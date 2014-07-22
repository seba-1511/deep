"""
MNIST Data Class
"""

import os
import gzip
import cPickle
from dataset import DataSet


class MNIST(DataSet):
    """ MNIST dataset from <link> """

    def __init__(self):

        data = self.load()
        super(MNIST, self).__init__(data)

    def load(self):

        data_dir = os.path.dirname(__file__) + "/data/"
        f = gzip.open(data_dir + "mnist.pkl.gz")
        data = cPickle.load(f)
        f.close()

        return data

    def plot_classes(self):
        """ plot mnist digits 1-9 """

        # TODO: plot mnist digits 1-9

        raise NotImplementedError


MNIST()
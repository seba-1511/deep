"""
MNIST Data Class
"""

import os
import gzip
import numpy
import cPickle
import matplotlib
import matplotlib.pyplot as plt
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

        fig = plt.figure()

        for i in range(1,10):

            ax = fig.add_subplot(3, 3, i)

            ith_digit = self.train_x[self.train_y == i][0]
            ith_digit = ith_digit.reshape(28, 28)

            ax.matshow(ith_digit, cmap=matplotlib.cm.binary)

            plt.xticks(numpy.array([]))
            plt.yticks(numpy.array([]))

        plt.show()
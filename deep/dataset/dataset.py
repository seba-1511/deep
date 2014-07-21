"""
Abstract Data Model
"""

import numpy as np


class DataSet(object):
    """  """

    # TODO: init that calls subclass load()

    def load(self):
        """ load data from pickled file """

        #raise NotImplementedError

    def reshape(self, set):
        """ reshape x and y sets """

        set_x, set_y = set

        set_x = self.reshape_x(set_x)
        set_y = self.reshape_y(set_y)

        return set_x, set_y

    def reshape_x(self, set_x):
        """ convert 1d vectors to 2d vectors """

        shape = list(set_x.shape)
        shape.append(1)

        return set_x.reshape(shape)

    def reshape_y(self, set_y):
        """ convert numeric labels to binary vectors """

        shape = (len(set_y), np.max(set_y)+1)
        bin_y = np.zeros(shape)
        for bin, y in zip(bin_y, set_y):
            bin[y] = 1

        shape = list(shape)
        shape.append(1)
        return bin_y.reshape(shape)

    def plot_classes(self):
        """ plot one example of each class """

        raise NotImplementedError

d = DataSet()
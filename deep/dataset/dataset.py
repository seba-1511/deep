"""
Abstract Data Model
"""

import numpy as np


class DataSet(object):
    """ abstract dataset class """

    def __init__(self, data):

        # TODO: make dataset logic more flexible. Include unsupervised datasets

        assert isinstance(data, tuple)
        assert len(data) >= 1, "dataset should contain atleast one tuple"
        assert len(data) <= 3, "dataset should contain 3 or less tuples"

        if len(data) >= 1:

            self.train_x, self.train_y = data[0]
            self.train_bin_y = self.reshape_y(self.train_y)

        if len(data) >= 2:

            self.valid_x, self.valid_y = data[1]
            self.valid_bin_y = self.reshape_y(self.valid_y)

        if len(data) == 3:

            self.test_x, self.test_y = data[2]
            self.test_bin_y = self.reshape_y(self.test_y)

    def load(self):
        """ does not implement load """

        raise NotImplementedError("DataSet does not implement load")

    def reshape_y(self, set_y):
        """ convert numeric labels to binary vectors """

        shape = (len(set_y), np.max(set_y)+1)
        bin_y = np.zeros(shape)
        for bin, y in zip(bin_y, set_y):
            bin[y] = 1

        return bin_y

    def plot_classes(self):
        """ plot one example of each class """

        raise NotImplementedError
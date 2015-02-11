# -*- coding: utf-8 -*-
"""
    deep.augmentation.base
    ----------------------

    Implements various types of data augmentation.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

from abc import abstractmethod

import numpy as np
from scipy.misc import imresize


#: unit test to check that augmentations return
#: a 2d numpy array


class Augmentation(object):

    def __init__(self, X):
        self.X = X

    def __iter__(self):
        return self

    def next(self):
        return self.X

    @property
    def shape(self):
        return self.X.shape

    def __len__(self):
        return len(self.X)


class AugmentationSequence(Augmentation):

    def __init__(self, X, augmentations):
        super(AugmentationSequence, self).__init__(X)
        self.augmentations = augmentations

    def next(self):
        X = self.X
        for augmentation in self.augmentations:
            X = augmentation(X)
        return X

    def shape(self):
        raise NotImplementedError


class RandomPatches(Augmentation):

    def __init__(self, X, patch_size):
        super(RandomPatches, self).__init__(X)
        self.patch_size = patch_size

    def next(self):
        n_samples = len(self.X)

        X = []
        for x in self.X:

            if x.ndim == 1:
                size = int(np.sqrt(len(x)))
                x = x.reshape((size, size))

            height, width = x.shape
            height_offset = np.random.randint(height - self.patch_size + 1)
            width_offset = np.random.randint(width - self.patch_size + 1)

            X.append(x[height_offset:height_offset+self.patch_size,
                     width_offset:width_offset+self.patch_size])

        return np.asarray(X).reshape(n_samples, -1)

    @property
    def shape(self):
        return self.X.shape[0], self.patch_size ** 2


def resize_shorter_side(examples, new_size_low, new_size_high):
    """

    :param examples: list of 2d numpy arrays
    :param new_dim: dimension of new images
    """

    #: how to include random new size
    #: make new_dim a tuple
    #: put this in a separate function

    X = []
    for x in examples:

        new_size = np.random.randint(new_size_low, new_size_high)
        height, width = x.shape

        #: added .0001 because certain images get resized
        #: to slightly smaller dimension than desired
        #: is scipy dropping figs on the multiplication?
        size = float(new_size) / min(height, width) + .0001
        X.append(imresize(x, size))
    return X


def rotate_images_90(examples):
    """

    :param examples: list of 2d numpy arrays
    """

    X = []
    for x in examples:
        rotations = np.random.randint(4)
        X.append(np.rot90(x, rotations))
    return np.asarray(X)
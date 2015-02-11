# -*- coding: utf-8 -*-
"""
    deep.augmentation.base
    ----------------------

    Implements various types of data augmentation.

    :references: pylearn2 (corruptions module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np

from scipy.misc import imresize


#: unit test to check that augmentations return
#: a 2d numpy array


class Augmentation(object):

    def __call__(self, X):
        return self.X


class AugmentationSequence(Augmentation):

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, X):
        for augmentation in self.augmentations:
            X = augmentation(X)
        return X


class RandomPatches(Augmentation):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, X):
        n_samples = len(X)

        patches = []
        for x in X:

            if x.ndim == 1:
                size = int(np.sqrt(len(x)))
                x = x.reshape((size, size))

            height, width = x.shape
            height_offset = np.random.randint(height - self.patch_size + 1)
            width_offset = np.random.randint(width - self.patch_size + 1)

            patches.append(x[height_offset:height_offset+self.patch_size,
                     width_offset:width_offset+self.patch_size])

        return np.asarray(patches).reshape(n_samples, -1)


class RandomRotation90(Augmentation):
    """

    :param examples: list of 2d numpy arrays
    """

    def __call__(self, X):
        n_samples = len(X)

        rotated = []
        for x in X:

            if x.ndim == 1:
                size = int(np.sqrt(len(x)))
                x = x.reshape((size, size))

            rotations = np.random.randint(4)
            rotated.append(np.rot90(x, rotations))
        return np.asarray(rotated).reshape(n_samples, -1)


#: this needs to be followed by RandomPatches
class RandomResize(Augmentation):
    """

    :param examples: list of 2d numpy arrays
    :param new_dim: dimension of new images
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, X):
        #: how to include random new size
        #: make new_dim a tuple
        #: put this in a separate function

        A = []
        for x in X:

            new_size = np.random.randint(self.low, self.high)
            height, width = x.shape

            #: added .0001 because certain images get resized
            #: to slightly smaller dimension than desired
            #: is scipy dropping figs on the multiplication?
            size = float(new_size) / min(height, width) + .0001
            A.append(imresize(x, size))
        return A

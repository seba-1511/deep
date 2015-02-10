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

    @abstractmethod
    def __call__(self, dataset):
        """"""

class AugmentationSequence(Augmentation):

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, dataset):
        for augmentation in self.augmentations:
            dataset = augmentation(dataset)
        return dataset


class RandomPatches(Augmentation):

    def __init__(self, patch_size):
        self.patch_size = patch_size

    @property
    def output_size(self):
        return self.patch_size ** 2

    def __call__(self, dataset):
        n_samples = len(dataset)

        if dataset.ndim == 2:
            dim = int(np.sqrt(dataset.shape[1]))
            dataset = dataset.reshape(-1, dim, dim)

        X = []
        for x in dataset:
            height, width = x.shape
            height_offset = np.random.randint(height - self.patch_size + 1)
            width_offset = np.random.randint(width - self.patch_size + 1)

            X.append(x[height_offset:height_offset+self.patch_size,
                     width_offset:width_offset+self.patch_size])
        return np.asarray(X).reshape(n_samples, -1)


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


def crop_random_patch(examples, patch_size):
    """

    :param examples: list of 2d numpy arrays
    :param patch_size: dimension of the patches
    """

    X = []
    for x in examples:
        height, width = x.shape
        height_offset = np.random.randint(height - patch_size + 1)
        width_offset = np.random.randint(width - patch_size + 1)

        X.append(x[height_offset:height_offset+patch_size,
                 width_offset:width_offset+patch_size])
    return np.asarray(X)


def rotate_images_90(examples):
    """

    :param examples: list of 2d numpy arrays
    """

    X = []
    for x in examples:
        rotations = np.random.randint(4)
        X.append(np.rot90(x, rotations))
    return np.asarray(X)
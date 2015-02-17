# -*- coding: utf-8 -*-
"""
    deep.autoencoders.stacked
    ------------------------

    Implements a stacked autoencoders.

    :references: pylearn2 (mlp module)

    :copyright: (c) 2014 by Gabriel Pereyra.
    :license: BSD, see LICENSE for more details.
"""


class SAE(object):

    #: add finetuning and score function

    def __init__(self, autoencoders=None):
        self.autoencoders = autoencoders

    def transform(self, X):
        for autoencoder in self.autoencoders:
            X = autoencoder.transform(X)
        return X

    def inverse_transform(self, X):
        for autoencoder in self.autoencoders[::-1]:
            X = autoencoder.inverse_transform(X)
        return X

    def fit(self, X):
        for autoencoder in self.autoencoders:
            X = autoencoder.fit_transform(X)

            from sklearn.preprocessing import MinMaxScaler
            #X = MinMaxScaler().fit_transform(X)

        return self

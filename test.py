print 'Loading Planktons...'
from deep.datasets.load import load_mnist
train, valid, test = load_mnist()
X, y = train
X_test, y_test = test

#: Only do that for MNIST
X = [i.reshape(28, 28) for i in X]
X_test = [i.reshape(28, 28) for i in X_test]

from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1)


import numpy as np
from deep.augmentation import Reshape, RandomPatch, HorizontalReflection
X_test = Reshape(26).fit_transform(X_test)
X_valid = Reshape(26).fit_transform(X_valid)

#: Augment data
X_patch = Reshape(28).fit_transform(X)
X_patch = RandomPatch(26).fit_transform(X_patch)
X = Reshape(26).fit_transform(X)
X_reflec = HorizontalReflection().fit_transform(X)

#: Merge the augmentations
X = np.vstack((X, X_patch, X_reflec))
y = np.tile(y, 3)

#: Standardize data
X = np.vstack((X, X_valid, X_test))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

#: Retrieve standardized
X_test = X[-len(X_test):]
X_valid = X[-(len(X_test) + len(X_valid)):-len(X_test)]
X = X[:-(len(X_test) + len(X_valid))]

print 'Defining Net...'
from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
from deep.activations.base import RectifiedLinear, Softmax
from deep.corruptions import Dropout
layers = [
    Layer(500, RectifiedLinear()),
    Layer(121, Softmax(), Dropout(.5)),
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.regularizers import L2
from deep.fit import Iterative
from deep.plot.base import plot_training
nn = NN(layers, .01, Momentum(.9), fit=Iterative(10), regularize=L2(.0005))
nn.fit(X, y, X_valid, y_valid)
print nn.score(X_test, y_test)
plot_training(nn, 'training')

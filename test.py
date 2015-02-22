print 'Loading Planktons...'
from deep.datasets.load import load_mnist
train, valid, test = load_mnist()
X, y = train
X_test, y_test = test

# from deep.augmentation import Reshape
# X = Reshape(48).fit_transform(X)
# X_test = Reshape(48).fit_transform(X_test)

import numpy as np
X = np.vstack((X, X_test))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print 'Defining Net...'
X_test = X[-len(X_test):]
X = X[:-len(X_test)]

from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
from deep.activations.base import RectifiedLinear, Softmax
from deep.corruptions import Dropout
layers = [
    Layer(500, RectifiedLinear()),
    Layer(121, Softmax(), Dropout(.5))
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.regularizers import L2
from deep.fit import Iterative
from deep.plot.base import plot_training
nn = NN(layers, .01, Momentum(.9), fit=Iterative(15), regularize=L2(.0005))
nn.fit(X, y)
plot_training(nn, 'General')


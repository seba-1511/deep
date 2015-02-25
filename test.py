
print 'Loading Planktons...'
import numpy as np
from deep.datasets.load import load_mnist
from deep.augmentation import Reshape, RandomPatch
from sklearn.preprocessing import StandardScaler
train, valid, test = load_mnist()

X, y = train
X_test, y_test = test

X = np.vstack((X, X_test))

X = StandardScaler().fit_transform(X)

X_test = X[-len(X_test):]
X = X[:-len(X_test)]


X = np.array([x.reshape(28, 28) for x in X])
X_test = np.array([x.reshape(28, 28) for x in X_test])

X = Reshape(26).fit_transform(X)
X_test = Reshape(26).fit_transform(X_test)


print 'Defining Net...'
from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
from deep.activations.base import RectifiedLinear, Softmax
from deep.corruptions import Dropout
layers = [
    Layer(500, RectifiedLinear()),
    Layer(10, Softmax(), Dropout(.5))
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.regularizers import L2
from deep.fit import Iterative
from deep.plot.base import plot_training
nn = NN(layers, .01, Momentum(.9), fit=Iterative(10), regularize=L2(.0005))
        # fixed_augmentation=[Reshape(26), RandomPatch(26)])
nn.fit(X, y)
print nn.score(X_test, y_test)
plot_training(nn, 'General')


print 'Loading Planktons...'
from deep.datasets.load import load_mnist
train, valid, test = load_mnist()
X, y = train
X_test, y_test = test

X = [i.reshape(28, 28) for i in X]
X_test = [i.reshape(28, 28) for i in X_test]


from deep.augmentation import Reshape, RandomPatch
X_patch = Reshape(28).fit_transform(X)
X_test = Reshape(26).fit_transform(X_test)
X = Reshape(26).fit_transform(X)
X_patch = RandomPatch(26).fit_transform(X_patch)


import numpy as np
X = np.vstack((X, X_patch))
y = np.append(y, y)
X = np.vstack((X, X_test))
# from deep.augmentation import Reshape
# X = Reshape(26).fit_transform(orX)
# X_test = Reshape(26).fit_transform(X_test)

# import numpy as np
# X = np.vstack((X, X_test))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print 'Defining Net...'
X_test = X[-len(X_test):]
X = X[:-len(X_test)]

from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
from deep.activations.base import RectifiedLinear, Softmax
from deep.corruptions import Dropout
rec = Layer(500, RectifiedLinear())
layers = [
    rec,
    Layer(121, Softmax(), Dropout(.5)),
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.regularizers import L2
from deep.fit import Iterative
from deep.plot.base import plot_training
nn = NN(layers, .01, Momentum(.9), fit=Iterative(10), regularize=L2(.0005))
nn.fit(X, y)
print nn.score(X_test, y_test)
import pdb; pdb.set_trace()

plot_training(nn, 'General')

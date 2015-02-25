# import collections


# def augment_data(X, y):
#     if not isinstance(self.fixed_augmentation, collections.Iterable):
#         X = self.fixed_augmentation.fit_transform(X)
#         return (X, y)
#     X_new = np.concatenate([augment(X)
#                             for augment in self.fixed_augmentation])
#     y = np.tile(y, len(self.fixed_augmentation))

#     from sklearn.preprocessing import StandardScaler
#     #: clean to be able to use any scaler
#     X_new = StandardScaler().fit_transform(X_new)
#     return (X_new, y)


# print 'Loading Planktons...'
# import numpy as np
# from deep.datasets.load import load_mnist
# from deep.augmentation import Reshape, RandomPatch, Resize, CenterPatch
# from sklearn.preprocessing import StandardScaler
# train, valid, test = load_mnist()
# X, y = train
# X_test, y_test = test

# X = X.reshape(-1, 28, 28)
# X_test = X.reshape(-1, 28, 28)

# X = Reshape(28).fit_transform(X)
# X_test = Reshape(28).fit_transform(X_test)

# X = np.vstack((X, X_test))
# X = StandardScaler().fit_transform(X)

# X_test = X[-len(X_test):]
# X = X[:-len(X_test)]

# from sklearn.cross_validation import train_test_split
# X, X_valid, y, y_valid = train_test_split(X, y, test_size=.01)

# print 'Defining Net...'
# from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
# from deep.activations.base import RectifiedLinear, Softmax
# from deep.corruptions import Dropout
# layers = [
#     Layer(500, RectifiedLinear()),
#     Layer(10, Softmax())
# ]

# print 'Learning...'
# from deep.models import NN
# from deep.updates import Momentum
# from deep.regularizers import L2
# from deep.fit import Iterative
# from deep.plot.base import plot_training
# nn = NN(layers, .01, Momentum(.9), fit=Iterative(10), regularize=L2(.0005))
# nn.fit(X, y, X_valid, y_valid)
# print nn.score(X_test, y_test)
# plot_training(nn, 'General')


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
from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y, test_size=.01)

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

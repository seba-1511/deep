from deep.datasets.load import load_plankton
X, y = load_plankton()

from deep.augmentation import Reshape
size = 28
X = Reshape(size).transform(X)

import numpy as np

def augment(X):
    X = X.reshape(-1, size, size)
    X_45 = np.rot90(X.T, 1).T.reshape(-1, size**2)
    X_90 = np.rot90(X.T, 2).T.reshape(-1, size**2)
    X_180 = np.rot90(X.T, 3).T.reshape(-1, size**2)

    X_lr = np.fliplr(X)
    X_45_lr = np.rot90(X_lr.T, 1).T.reshape(-1, size**2)
    X_90_lr = np.rot90(X_lr.T, 2).T.reshape(-1, size**2)
    X_180_lr = np.rot90(X_lr.T, 3).T.reshape(-1, size**2)

    X = X.reshape(-1, size**2)
    X_lr = X_lr.reshape(-1, size**2)

    return X, X_45, X_90, X_180, X_lr, X_45_lr, X_90_lr, X_180_lr

X = np.vstack(augment(X))
y = np.hstack((y, y, y, y, y, y, y, y))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y)

from deep.datasets import SupervisedData
train = SupervisedData((X, y))
valid = SupervisedData((X_valid, y_valid))

from deep.layers.base import Layer, PreConv, PostConv, Pooling, ConvolutionLayer
from deep.activations import RectifiedLinear, Softmax
from deep.corruptions import Dropout
layers = [
    PreConv(),
    ConvolutionLayer(64, 7, 1, RectifiedLinear()),
    Pooling(5, 2),
    ConvolutionLayer(64, 5, 1, RectifiedLinear()),
    Pooling(3, 2),
    PostConv(),
    Layer(1000, RectifiedLinear()),
    Layer(121, Softmax())
]

from deep.networks import NN
from deep.updates import Momentum
from deep.fit import EarlyStopping
nn = NN(layers, learning_rate=.01, update=Momentum(.9), fit=EarlyStopping(valid))
nn.fit(train)

predictions = [nn.predict_proba(X) for X in augment(X_valid)]
predictions = np.asarray(predictions)
prediction = np.argmax(np.mean(predictions, axis=0), axis=1)
print np.mean(prediction == y_valid)
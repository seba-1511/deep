"""
Temporary file for messing around
"""
from deep.dataset.kaggle_decoding import KaggleDecoding
from deep.model import mlp
from deep.train import train

k = KaggleDecoding()

layer1 = mlp.SigmoidLayer(375, 3)
layer2 = mlp.SigmoidLayer(3, 2)
layers = [layer1, layer2]

mlp = mlp.MLP(layers)

data = k.load()

train_x, train_y = k.train_set


import sklearn.linear_model
import numpy as np

clf = sklearn.linear_model.LogisticRegression()

print train_x.shape

train_x = train_x[:, 277, :]
train_x -= train_x.mean(0)
train_x = np.nan_to_num(train_x / train_x.std(0))


train.sgd((train_x, train_y), mlp)
train.score((train_x, train_y), mlp)

#clf.fit(train_x, train_y)
#print clf.score(train_x, train_y)


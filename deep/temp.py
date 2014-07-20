"""
Temporary file for messing around
"""

import load
import ae
import mlp
import train

data = load.kaggle_decoding()

layer1 = mlp.SigmoidLayer(375, 20)
layer2 = mlp.SigmoidLayer(20, 1)
layers = [layer1, layer2]

mlp = mlp.MLP(layers)

train_x, train_y = data[0]

train_x = train_x[:, 179,:].reshape(594, 375, 1)
train_y = train_y.reshape(-1, 1)

train.sgd((train_x, train_y), mlp)

train.score((train_x, train_y), mlp)

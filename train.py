"""
Model Parameter Selection
"""

import numpy as np
import mlp
import load_data

def sgd(data, model):

    train_x, train_y = data[0]
    valid_x, valid_y = data[1]

    epochs = 10

    for epoch in range(epochs):

        for x, y in zip(train_x, train_y):

            model.bprop(x, y)

        correct = 0.0

        for x, y in zip(valid_x, valid_y):

            if(np.argmax(model.fprop(x)) == np.argmax(y)):

                correct += 1.0

        print correct / len(valid_x)

data = load_data.mnist()
data = load_data.reshape(data)

layer1 = mlp.SigmoidLayer(784, 30)
layer2 = mlp.SigmoidLayer(30, 10)
layers = [layer1, layer2]

mlp = mlp.MLP(layers)

sgd(data, mlp)


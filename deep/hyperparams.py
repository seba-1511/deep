from deep.activations.base import Sigmoid, Softmax

# default
layer_sizes = [100, 100]
activations = [Sigmoid, Softmax]
batch_size = 100
learning_rate = .1
n_iter = 10
n_hidden = 100

# add hyperparams used to recreate experiment results
from deep.train.train import score
from deep.train.train import bgd
from deep.model.mlp import SigmoidLayer
from deep.dataset.mnist import MNIST

m = MNIST()


s = SigmoidLayer(784, 10)




bgd(m, s)
score(m, s)
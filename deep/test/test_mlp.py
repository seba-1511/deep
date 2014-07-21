from deep.model.mlp import SigmoidLayer
from deep.model.mlp import MLP

# TODO: unit test mlp

sl = SigmoidLayer(10, 10)
mlp = MLP(sl)
from deep.datasets.load import load_plankton
X, y = load_plankton()

from deep.augmentation import Reshape
X = Reshape(28).fit_transform(X)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from deep.layers import ConvolutionLayer, PreConv, PostConv, Layer, Pooling
from deep.activations import RectifiedLinear, Softmax
from deep.corruptions import Gaussian
layers = [
    PreConv(),
    ConvolutionLayer(32, 7, RectifiedLinear()),
    Pooling(5, 2),
    ConvolutionLayer(64, 5, RectifiedLinear()),
    Pooling(3, 2),
    PostConv(),
    Layer(1000, RectifiedLinear()),
    Layer(121, Softmax(), Gaussian(.25))
]

from deep.models import NN
from deep.updates import Momentum
from deep.fit import EarlyStopping
from deep.regularizers import L2
nn = NN(layers, .01, Momentum(.9), EarlyStopping(batch_size=128), regularize=L2(.0005))
nn.fit(X, y)

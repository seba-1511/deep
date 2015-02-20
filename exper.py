print 'Loading Planktons...'
from deep.datasets.load import load_plankton
X, y = load_plankton()

from deep.augmentation import Reshape
X = Reshape(28).fit_transform(X)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print 'Defining Net...'
from deep.layers import ConvolutionLayer, PreConv, PostConv, Layer, Pooling
from deep.activations import RectifiedLinear, Softmax
from deep.corruptions import Gaussian
layers = [
    PreConv(),
    ConvolutionLayer(94, 3, RectifiedLinear()),
    Pooling(4, 2),
    ConvolutionLayer(94, 3, RectifiedLinear()),
    Pooling(3, 2),
    PostConv(),
    Layer(1048, RectifiedLinear()),
    Layer(121, Softmax(), Gaussian(.25))
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.fit import EarlyStopping
from deep.regularizers import L2
nn = NN(layers, .01, Momentum(.9), EarlyStopping(batch_size=128), regularize=L2(.005))
nn.fit(X, y)

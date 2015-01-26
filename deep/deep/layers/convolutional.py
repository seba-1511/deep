import theano
from theano.tensor.signal import downsample

import numpy as np


def image_square(n):
    n_sqrt = np.sqrt(n)
    assert n_sqrt ** 2 == n
    return n_sqrt

class ConvolutionLayer(object):

    def __init__(self, n_filters, filter_size, batch_size, n_features):
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.image_sqrt = image_square(n_features)
        shape = (n_filters, 1, filter_size, filter_size)
        range = np.sqrt(24. / sum(shape))
        W = np.random.uniform(low=-range, high=range, size=shape)
        self.W = theano.shared(np.asarray(W, dtype='float32'))
        self.b = theano.shared(np.zeros(shape[0]))

    @property
    def output_size(self):
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)

        new_dim = self.image_sqrt - self.filter_size + 1

        return (new_dim / 2) ** 2 * self.n_filters

    def transform(self, x):
        x = x.reshape((self.batch_size, 1, 28, 28))

        print x
        print self.W

        conv_out = T.nnet.conv2d(x, self.W, subsample=(2, 2)) #pooled_out = downsample.max_pool_2d(conv_out, (2, 2)) difference?
        return T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)

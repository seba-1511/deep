import gzip
import cPickle
import numpy
from scipy.io import loadmat

def load_mnist():
    """ load mnist data from ? """ #TODO

    f = gzip.open("data/mnist.pkl.gz")
    data = cPickle.load(f)
    f.close()

    return data


def load_kaggle_decoding():
    """ load kaggle decoding data from ? """ #TODO

    train_subject_range = range(1,2)
    valid_subject_range = range(11,12)
    test_subject_range = range(17,18)

    train_x = []
    train_y = []

    for subject in train_subject_range:
        filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        train_x.append(data['X'])
        train_y.append(data['y'])

    train_x = numpy.vstack(train_x)
    train_y = numpy.concatenate(train_y)

    valid_x = []
    valid_y = []

    for subject in valid_subject_range:
        filename = 'data/kaggle_decoding/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        valid_x.append(data['X'])
        valid_y.append(data['y'])

    valid_x = numpy.vstack(valid_x)
    valid_x = numpy.concatenate(valid_y)

    test_x = []
    test_y = []

    for subject in test_subject_range:
        filename = 'data/kaggle_decoding/test_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)

        test_x.append(data['X'])
        test_y.append(data['Id'])

    test_x = numpy.vstack(test_x)
    test_y = numpy.concatenate(test_y)

    return ((train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y))


import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import gzip

def create_y_vector(y):
    y_vector = np.zeros(10)
    y_vector[y] = 1
    return y_vector

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))


if __name__ == "__main__":

    # random seed
    np.random.seed(1)

    # load training data
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)

    # take small subset for testing
    train_x = training_data[0][:100]
    train_y = training_data[1][:100]

    # convert y labels to vectors
    train_y = [create_y_vector(y) for y in train_y]

    # initialize network
    size = [784, 30, 10]
    weights = [np.random.randn(x,y) for x, y in zip(size[1:], size[:-1])]
    biases = [np.random.randn]
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import gzip

def create_y_vector(y):
    y_vector = np.zeros((10,1))
    y_vector[y] = 1
    return y_vector

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feedforward(weights, a):

    # for each layer
    for w in weights:
        a = sigmoid(np.dot(w, a))
        # add 1 for bias
        a = np.insert(a,0,1)
    # remove 1 from final activation
    return a[1:]

def cost(weights, set_x, set_y):

    # sum of the square difference of output and label vector
    return np.sum([(feedforward(weights, x) - train_y)**2 for
                   x, y in zip(set_x, set_y)])

if __name__ == "__main__":

    # random seed
    np.random.seed(1)

    # load training data
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)

    # take small subset for testing
    train_x = training_data[0][:100]
    train_y = training_data[1][:100]

    # append 0's to x's and
    # convert y labels to vectors
    train_x = np.column_stack((np.ones(len(train_x)),train_x))
    train_y = [create_y_vector(y) for y in train_y]

    # initialize network
    size = [784, 30, 10]
    weights = [np.random.randn(x,y+1) for x, y in zip(size[1:], size[:-1])]

    # feedforward
    print feedforward(weights, train_x[0])

    print cost(weights, train_x, train_y)
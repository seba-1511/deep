# path to modules
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

import numpy as np
import cPickle
import gzip

def vectorize_label(y):
    e = np.zeros((10,1))
    e[y] = 1.0
    return e

def load_mnist_data():

    f = gzip.open('data/mnist.pkl.gz','rb')
    train_data, valid_data, test_data = cPickle.load(f)
    f.close()

    return (train_data, valid_data, test_data)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_w = np.random.randn(hidden_size, input_size)
        self.hidden_b = np.random.randn(hidden_size, 1)
        self.visible_w = np.random.randn(output_size, hidden_size)
        self.visible_b = np.random.randn(output_size, 1)

    def get_hidden_values(self, input):
        input = input.reshape(len(input), 1)
        hidden_z = np.dot(self.hidden_w, input) + self.hidden_b
        hidden_a = sigmoid(hidden_z)
        return (hidden_z, hidden_a)

    def get_output_values(self, hidden_values):
        visible_z = np.dot(self.visible_w, hidden_values) + self.visible_b
        visible_a = sigmoid(visible_z)
        return (visible_z, visible_a)

    def get_cost(self, x, y):
        hidden_z, hidden_a = self.get_hidden_values(x)
        visible_z, visible_a = self.get_output_values(hidden_a)
        return np.mean((y - visible_a)**2)

    def predict(self, x):
        hidden_z, hidden_a = self.get_hidden_values(x)
        visible_z, visible_a = self.get_output_values(hidden_a)
        return np.argmax(visible_a)

    def update(self, x, y, lr=0.1):

        hidden_z, hidden_a = self.get_hidden_values(x)
        visible_z, visible_a = self.get_output_values(hidden_a)

        visible_d = -(y - visible_a) * sigmoid_prime(visible_z)
        hidden_d = np.dot(self.visible_w.T, visible_d) * sigmoid_prime(hidden_z)

        visible_g = np.dot(visible_d, hidden_a.T)
        hidden_g = np.dot(hidden_d, x.reshape(len(x),1).T)

        cost = self.get_cost(x, y)
        self.visible_w -= lr * visible_g * cost
        self.visible_b -= lr * visible_d * cost

        self.hidden_w -= lr * hidden_g * cost
        self.hidden_b -= lr * hidden_d * cost

    def train(self, train_x, train_y, lr=3, epochs=1000):

        for epoch in range(epochs):

            total_cost = 0

            for x, y in zip(train_x, train_y):
                total_cost += self.get_cost(x, y)
                self.update(x, y, lr)

            print 'total cost:', total_cost, 'valid %:', self.score(valid_x, valid_y)

    def score(self, valid_x, valid_y):

        correct = 0.

        for x, y in zip(valid_x, valid_y):

            if self.predict(x) == np.argmax(y):

                correct += 1

        return correct / len(valid_x)

# load data
data = load_mnist_data()
train_x, train_y = data[0]
train_y = [vectorize_label(y) for y in train_y]
valid_x, valid_y = data[1]
valid_y = [vectorize_label(y) for y in valid_y]

net = NeuralNetwork(784, 30, 10)

net.train(train_x, train_y)

print net.score(valid_x, valid_y)
# path to modules
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")

from sklearn import datasets
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class neural_network:

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_w = np.random.random((hidden_size, input_size))
        self.hidden_b = np.random.random((hidden_size, 1))

        self.visible_w = np.random.random((output_size, hidden_size))
        self.visible_b = np.random.random((output_size, 1))

    def back_prop(self, x, y, learning_rate):

        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)

        hidden_z = np.dot(self.hidden_w, x) + self.hidden_b
        hidden_a = sigmoid(hidden_z)

        visible_z = np.dot(self.visible_w, hidden_a) + self.visible_b
        visible_a = sigmoid(visible_z)

        visible_d = (y - visible_a) * -sigmoid_prime(visible_z)
        hidden_d = np.dot(self.visible_w.T, visible_d) * -sigmoid_prime(hidden_z)

        visible_g = np.dot(visible_d, hidden_a.T)
        hidden_g = np.dot(hidden_d, x.T)

        self.hidden_w -= learning_rate * hidden_g
        self.hidden_b -= learning_rate * hidden_d

        self.visible_w -= learning_rate * visible_g
        self.visible_b -= learning_rate * visible_d

    def cost(self, x, y):

        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)

        hidden_z = np.dot(self.hidden_w, x) + self.hidden_b
        hidden_a = sigmoid(hidden_z)

        visible_z = np.dot(self.visible_w, hidden_a) + self.visible_b
        visible_a = sigmoid(visible_z)

        return np.mean((y - visible_a)**2)

    def predict(self, x):

        x = x.reshape(len(x), 1)

        hidden_z = np.dot(self.hidden_w, x) + self.hidden_b
        hidden_a = sigmoid(hidden_z)

        visible_z = np.dot(self.visible_w, hidden_a) + self.visible_b
        visible_a = sigmoid(visible_z)

        return np.argmax(visible_a)

    def fit(self, train_x, train_y, learning_rate=0.1, epochs=5):

        for epoch in range(epochs):

            total_cost = 0

            for x, y in zip(train_x, train_y):
                total_cost += self.cost(x, y)
                self.back_prop(x, y, learning_rate)

            print total_cost

    def score(self, valid_x, valid_y):

        correct = 0.0

        for x, y in zip(valid_x, valid_y):
            if self.predict(x) == np.argmax(y):
                correct += 1

        return correct / len(valid_x)

def vectorized_target(y):
    e = np.zeros((3,1))
    e[y] = 1.0
    return e

if __name__ == "__main__":

    iris = datasets.load_iris()
    train_x = iris.data
    train_y = [vectorized_target(y).reshape(3) for y in iris.target]

    combined = zip(train_x, train_y)
    np.random.shuffle(combined)
    train_x, train_y = zip(*combined)

    net = neural_network(4,4,3)
    net.fit(train_x, train_y)
    print net.score(train_x, train_y)
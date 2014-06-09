import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

if __name__ == "__main__":

    # 100 training examples of 2 vars
    train_x = np.random.randn(100,2)
    train_y = []

    # generate ys
    for x,y in train_x:
        if x + y > 0:
            train_y.append(1)
        else:
            train_y.append(0)

    # shift up and right 4
    train_x = train_x+4

    # initialize net
    sizes = [2,1]
    num_layers = len(sizes)
    bias = np.random.randn(1)
    weights = np.random.randn(2)

    for i in range(10):

        print "bias   :", bias
        print "weights:", weights

        # plot training examples
        for x, y in zip(train_x, train_y):
            x1, x2 = x
            if y == 1:
                plt.scatter(x1,x2,c='b')
            if y == 0:
                plt.scatter(x1,x2,c='r')

        # create line
        line = []
        for j in range(0,9):
            x1 = j
            x2 = (bias - weights[0] * x1)  / weights[1]
            line.append((x2))

        # plot line
        plt.plot(line)
        plt.axis([0,8,0,8])
        plt.show()

        print i
        print "example  :", train_x[i], "label:", train_y[i]

        update = weights - weights

        for i in range(len(train_x)):

            raw_activation = np.dot(weights, train_x[i])
            squashed_activation = sigmoid(raw_activation)
            error = train_y[i] - squashed_activation
            update += error * train_x[i]

        weights = weights + update * .01
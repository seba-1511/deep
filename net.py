import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def cost(w, x, y):

    return np.sum([(np.dot(x,w) - y)**2 for x,y in zip(train_x, y)])

def update(w, train_x, y):

    for x, y in zip(train_x, y):
        w = w + 0.1*(y - sigmoid(np.dot(x, w)))*x
    return w

if __name__ == "__main__":

    # seed and generate x and y
    np.random.seed(7)
    x = np.random.randn(10,2) + 2
    y = np.random.randint(0,2,10)

    # generate weights
    w = np.random.randn(3)

    # append ones to x
    train_x  = np.column_stack((np.ones(10), x))

    for i in range(100):
        w = update(w, train_x, y)
        print cost(w, train_x, y)

    # compute line and plot
    line = [-(w[0] + w[1] * i) / w[2]  for i in range(5)]
    plt.plot(line)

    # stack columns (since zip won't work) and plot
    d = np.column_stack((x,y))
    [plt.scatter(x1,x2, c=('r' if y == 1 else 'b')) for x1, x2, y in d]
    plt.show()


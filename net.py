import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    train_x = np.random.randn(100,2)
    train_y = []

    for x,y in train_x:
        if x + y > 0:
            train_y.append(1)
        else:
            train_y.append(0)

    for x, y in zip(train_x, train_y):

        x1, x2 = x

        if y == 1:
            plt.scatter(x1,x2,c='b')
        if y == 0:
            plt.scatter(x1,x2,c='r')

    plt.show()
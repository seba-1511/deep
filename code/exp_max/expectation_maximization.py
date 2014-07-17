import numpy as np
import matplotlib.pyplot as plt

def load_old_faithful_data():

    f = open("../../data/old_faithful.txt")
    lines = [line.split() for line in f.readlines()]
    array = np.array(lines, dtype=float)

    array -= array.mean(0)

    array /= np.std(array, axis=0)

    return array[:,1:]

class KM():

    def __init__(self, data, number_of_clusters):

        self.data = data
        self.k = number_of_clusters

        self.means = [[-1,1], [1,-1]]
        self.assignments = 0
        self.expectation_step()

    def expectation_step(self):

        distance = np.zeros((len(self.data), self.k))

        for i in range(self.k):
            distance[:, i] = np.sum((self.data - self.means[i]) ** 2, axis=1)

        r = np.argmin(distance, axis=1)

        assignments = np.zeros((len(self.data), self.k))

        for i in range(len(self.data)):

            assignments[i][r[i]] = 1

        self.assignments = assignments

    def maximization_step(self):

        for i in range(self.k):

            self.means[i] = np.sum(self.assignments[:,i].reshape(-1,1) * data, axis=0) \
                            / np.sum(self.assignments[:,i])

    def train(self):

        for i in range(10):

            self.expectation_step()
            self.maximization_step()

    def plot(self):

        for i in range(len(self.data)):

            if self.assignments[i][0] == 1:
                plt.scatter(data[i][0], data[i][1], c='b')
            else:
                plt.scatter(data[i][0], data[i][1], c='r')

        plt.scatter(self.means[0][0], self.means[0][1], c='b', marker='x')
        plt.scatter(self.means[1][0], self.means[1][1], c='r', marker='x')

        plt.show()

class EM():

    def __init__(self, data, number_of_clusters):

        self.data = data
        self.k = number_of_clusters
        self.n = data.shape[1]

        self.means = [[-1,1],[1,-1]]
        self.covs = np.array([np.eye(2) for i in range(2)])

        self.pi = np.random.random((self.n, 1))
        self.pi = self.pi / sum(self.pi)

    def log_likelihood(self):

        # TODO

        raise NotImplementedError

    def expectation_step(self):

        # TODO

        raise NotImplementedError

    def maximization_step(self):

        # TODO

        raise NotImplementedError

    def plot(self):

        [plt.scatter(x,y) for x,y in self.data]
        [plt.scatter(x,y, c='r') for x,y in self.means]

        # TODO: add standard deviation circles
        #http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib

        plt.show()

if __name__ == "__main__":

    data = load_old_faithful_data()

    km = KM(data, 2)
    km.plot()
    km.train()
    km.plot()
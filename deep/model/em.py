import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def load_old_faithful_data():

    f = open("../../data/old_faithful.txt")
    lines = [line.split() for line in f.readlines()]
    array = np.array(lines, dtype=float)

    array -= array.min(0)

    array /= np.std(array, axis=0)

    return array[:,1:]


class KM():

    def __init__(self, data, number_of_clusters):

        self.data = data
        self.k = number_of_clusters

        self.means = [[-1,1], [1,-1]]
        self.assignments = None
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

        self.means = [[-1,1],[1,-1]]
        self.covs = np.array([np.eye(2) for _ in range(2)])
        r = np.random.random(2)
        self.pis = r / np.sum(r)
        self.gamma = np.zeros((len(self.data), self.k))

    def expectation_step(self):

        self.gamma[:, 0] = scipy.stats.multivariate_normal.pdf(data, self.means[0], self.covs[0])
        self.gamma[:, 1] = scipy.stats.multivariate_normal.pdf(data, self.means[1], self.covs[1])
        self.gamma /= np.sum(self.gamma, axis=1).reshape(-1, 1)

    def maximization_step(self):

        self.means[0] = np.sum(self.data * self.gamma[:,0].reshape(-1, 1), axis=0)
        self.means[1] = np.sum(self.data * self.gamma[:,1].reshape(-1, 1), axis=0)
        self.means /= np.sum(self.gamma, axis=0)

        self.covs[0] = np.sum(self.gamma[:,0] * np.dot((data - self.means[0]),
                                                       (data - self.means[0]).T))
        self.covs[1] = np.sum(self.gamma[:,1] * np.dot((data - self.means[1]),
                                                       (data - self.means[1]).T))
        self.covs /= np.sum(self.gamma, axis=0)

        self.pis = np.sum(self.gamma, axis=0) / len(self.data)


    def plot(self):

        [plt.scatter(x,y) for x,y in self.data]
        [plt.scatter(x,y, c='r', marker='x') for x,y in self.means]

        # TODO: add standard deviation contours
        #http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib

        plt.show()


class CorEx():
    """
    Correlation Explanation Algorithm

    Simplifying assumptions
    - observed variables are binary
    - latent variables are binary

    """

    def __init__(self):

        # initialize ns x n matrix of discrete variables
        num_observed_samples = 1000
        num_observed_variables = 10
        observed_data_shape = (num_observed_samples, num_observed_variables)
        observed_data = np.random.randint(0, 2, observed_data_shape)

        # initialize m x k matrix of latent variables
        num_latent_variables = 2
        dim_latent_variables = 2
        latent_variable_shape = (num_latent_variables, dim_latent_variables)
        latent_variables = np.random.randint(0, 2,  latent_variable_shape)

        # initialize alpha i x j
        indicator_shape = (num_observed_variables, num_latent_variables)
        indicator_variable = np.zeros(indicator_shape)
        indicator_variable[:, 0] = 1

        # initialize p_y_x_sample

        #????????????????????????????????????????

    def update_marginals(self):





        raise NotImplementedError

    def marginal_mutual_information(self):

        raise NotImplementedError

    def update_alpha(self):

        raise NotImplementedError

    def update_p_y_given_x(self):

        raise NotImplementedError

    def train(self):

        raise NotImplementedError

corex = CorEx()

import practice
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def get_max_activation(hidden_weights):

    squared = hidden_weights**2
    sum = np.sum(squared, axis=1)
    root = np.sqrt(sum)

    return hidden_weights / root.reshape(len(root), 1)

def plot_max_activations(max_activations):

    fig = plt.figure()

    for x in range(6):
        for y in range(5):

            ax = fig.add_subplot(6, 5, 6*y+x)
            image = max_activations[x+y].reshape(28,28)
            ax.matshow(image, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()

def plot_reconstructed_input(train_x, train_y, net, filename):

    fig = plt.figure()

    for x in range(3):

        for y in range(3):

            ax = fig.add_subplot(3, 3, 3*x+y)

            _, hidden_a = net.get_hidden_values(train_x[train_y == 3*x+y][0])
            _, output = net.get_output_values(hidden_a)

            image = output.reshape(28, 28)
            ax.matshow(image, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.savefig(filename)

def print_digits_one_through_nine(train_x, train_y):

    fig = plt.figure()

    for x in range(3):

        for y in range(3):

            ax = fig.add_subplot(3, 3, 3*x+y)
            image = train_x[train_y == 3*x+y][0]
            image = image.reshape(28, 28)
            ax.matshow(image, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()


def add_noise_to_data(train_x):

    noise =  np.random.binomial(1, 0.1, train_x.shape)

    return abs(noise - train_x)


if __name__ == "__main__":

    data = practice.load_mnist_data()
    train_x, train_y = data[0]

    #print_digits_one_through_nine(train_x, train_y)
    #train_x = add_noise_to_data(train_x)
    #print_digits_one_through_nine(train_x, train_y)

    net = practice.NeuralNetwork(784, 50, 784)
    net.train(train_x, train_x, epochs=5)
    plot_reconstructed_input(train_x, train_y, net, 'mnist_digits.png')

    train_x = add_noise_to_data(train_x)

    net.train(train_x, train_x, epochs=5)
    plot_reconstructed_input(train_x, train_y, net, 'noisy_mnist_digits.png')
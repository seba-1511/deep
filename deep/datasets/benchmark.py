from sklearn.base import ClassifierMixin, TransformerMixin
from deep.datasets import load_mnist
from deep.datasets import load_digits


def benchmark_supervised_mnist(model):

    assert isinstance(model, ClassifierMixin)

    data = load_mnist()
    X_train, y_train = data[0]
    X_test, y_test = data[2]

    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def benchmark_unsupervised_mnist(model):

    data = load_mnist()
    X_train = data[0][0]
    X_test = data[2][0]

    print X_train.min(), X_train.max()

    model.fit(X_train)
    return model.score(X_test)


def benchmark_supervised_digits(model):

    assert isinstance(model, ClassifierMixin)

    data = load_digits()
    X, y = data

    X -= X.min()
    X /= X.max()

    X_train = X[:1500]
    y_train = y[:1500]
    X_test = X[1500:]
    y_test = y[1500:]

    model.fit(X_train, y_train)
    print model.score(X_test, y_test)


def benchmark_unsupervised_digits(model):

    data = load_digits()
    X = data[0]

    X -= X.min()
    X /= X.max()

    X_train = X[:1500]
    X_test = X[1500:]

    model.fit(X_train)
    model.score(X_test)

from deep.neural_network import TheanoMultiLayerPerceptron
from deep.neural_network import SigmoidLayer, SoftMaxLayer


from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
X, y = load_digits()

clf = LogisticRegression()
print learning_curve(clf, X, y)

print y.shape
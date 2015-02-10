from abc import ABCMeta
from sklearn.base import BaseEstimator


class LayeredModel(BaseEstimator):
    """A model parametrized by a list of layers"""

    __metaclass__ = ABCMeta

    #: layers = abstractproperty? (leads to syntax error in __len__)
    layers = None

    @property
    def params(self):
        """Collects the weight matrices and biases of each layer.
        :rtype : object
        """
        return [param for layer in self.layers for param in layer.params]

    @property
    def updates(self):
        """Collects the updates for each param in each layer."""
        rv = list()
        for param in self.params:
            cost = self._symbolic_cost(self.x, self.y)
            updates = self.update(cost, param, self.learning_rate)
            for update in updates:
                rv.append(update)
        return rv

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, item):
        return self.layers[item]

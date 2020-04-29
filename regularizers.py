from abc import ABCMeta, abstractmethod
import numpy as np


def _norm(n_xx):
    return np.clip(n_xx, 0, np.inf) / (np.clip(n_xx, 0, np.inf).sum(axis=0) + 1e-6)


class BaseRegularizer(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Apply regularization rule for likelihood function. """


class LDARegularizer(BaseRegularizer):

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def __call__(self, **kwargs):
        return self.alpha - 1.


class Regularizers():

    def __init__(self, regularizers: list = None):
        self.regularizers = regularizers

    def empty(self):
        return not self.regularizers

    def apply(self, n_xx, **kwargs):
        return _norm(n_xx + np.sum([reg(**kwargs) for reg in self.regularizers]))




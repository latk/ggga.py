import abc
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import typing as t
from .space import Space
from .util import ToJsonish


class SurrogateModel(ToJsonish, abc.ABC):
    space: Space

    @classmethod
    @abc.abstractmethod
    def estimate(
        cls, xs: np.ndarray, ys: np.ndarray, *,
        space: Space,
        rng: RandomState,
        prior: 'SurrogateModel',
    ) -> 'SurrogateModel':
        pass

    def predict(self, sample: list, *, return_std: bool = True):
        if return_std:
            mean, std = self.predict_a([sample], return_std=True)
            return mean[0], std[0]
        else:
            mean = self.predict_a([sample], return_std=False)
            return mean[0]

    def predict_a(self, multiple_samples: list, return_std: bool=True):
        return self.predict_transformed_a(
            [self.space.into_transformed(sample)
                for sample in multiple_samples],
            return_std=return_std,
        )

    @abc.abstractmethod
    def predict_transformed_a(self, X: t.Iterable, *, return_std: bool=True):
        pass

    @abc.abstractmethod
    def length_scales(self) -> np.ndarray:
        return np.array([1.0] * self.space.n_dims)

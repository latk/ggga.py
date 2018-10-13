import abc
import typing as t

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .space import Space
from .util import ToJsonish


class SurrogateModel(ToJsonish, abc.ABC):
    space: Space

    @classmethod
    @abc.abstractmethod
    def estimate(
        cls, mat_x: np.ndarray, vec_y: np.ndarray, *,
        space: Space,
        rng: RandomState,
        prior: t.Optional['SurrogateModel'],
        **kwargs,
    ) -> 'SurrogateModel':
        raise NotImplementedError

    def predict(
        self, vec_x: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[float, t.Optional[float]]:
        mean, std = self.predict_a([vec_x], return_std=return_std)
        if return_std:
            assert std is not None
            return mean[0], std[0]
        return mean[0], std

    def predict_a(
        self, mat_x: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        return self.predict_transformed_a(
            [self.space.into_transformed(sample)
                for sample in mat_x],
            return_std=return_std,
        )

    @abc.abstractmethod
    def predict_transformed_a(
        self, mat_x_transformed: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        raise NotImplementedError

    def length_scales(self) -> np.ndarray:
        return np.array([1.0] * self.space.n_dims)

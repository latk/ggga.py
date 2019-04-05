import abc
import typing as t

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .space import Space
from .util import ToJsonish


class SurrogateModel(ToJsonish, abc.ABC):
    """*interface* - A regression model to predict the value of points.
    This is used to guide the acquisition of new samples.

    Subclasses must override
    :meth:`estimate` and
    :meth:`predict_transformed_a`.

    Known implementations:

    .. autosummary::
        ggga.gpr.SurrogateModelGPR
        ggga.knn.SurrogateModelKNN
    """

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
        """Fit a new model to the given data.

        Parameters
        ----------
        mat_x:
        vec_y:
        space:
        rng:
        prior:
        **kwargs:
            Extra arguments for the concrete SurrogateModel class.
        """
        raise NotImplementedError

    def predict(
        self, vec_x: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[float, t.Optional[float]]:
        """
        Returns
        -------
        mean: float
        std: float or None
        """
        mean, std = self.predict_a([vec_x], return_std=return_std)
        if return_std:
            assert std is not None
            return mean[0], std[0]
        return mean[0], std

    def predict_a(
        self, mat_x: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        """
        Returns
        -------
        vec_mean: np.ndarray
        vec_std: np.ndarray or None
        """
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
        """Predict multiple values at the same time.

        Returns
        -------
        vec_mean : np.ndarray
        vec_std : np.ndarray or None
        """
        raise NotImplementedError

    def length_scales(self) -> np.ndarray:
        """Length scales for the paramters, estimated by the fitted model.

        Longer length scales indicate less relevant parameters.
        By default, the scale is 1.
        """
        return np.array([1.0] * self.space.n_dims)

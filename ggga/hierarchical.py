import typing as t

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .surrogate_model import SurrogateModel
from .space import Space
from .util import coerce_array


class SurrogateModelHierarchical(SurrogateModel):

    def __init__(
        self,
        base_model: SurrogateModel,
        detail_model: SurrogateModel,
        space: Space,
    ) -> None:
        self.base_model = base_model
        self.detail_model = detail_model
        self.space = space

    def to_jsonish(self):
        data = dict()
        data['model_class'] = type(self).__name__
        data['base_model'] = self.base_model.to_jsonish()
        data['detail_model'] = self.detail_model.to_jsonish()
        return data

    @classmethod
    def estimate(  # pylint: disable=arguments-differ
        cls, mat_x: np.ndarray, vec_y: np.ndarray, *,
        space: Space,
        rng: RandomState,
        prior: t.Optional[SurrogateModel],
        base_model_class: t.Type[SurrogateModel] = None,
        detail_model_class: t.Type[SurrogateModel] = None,
        **kwargs,
    ) -> 'SurrogateModelHierarchical':
        assert base_model_class is not None
        assert detail_model_class is not None

        if kwargs:
            raise TypeError(f"Unknown arguments: {sorted(kwargs)}")

        base_prior = None
        detail_prior = None
        if prior is not None:
            assert isinstance(prior, SurrogateModelHierarchical)
            base_prior = prior.base_model
            detail_prior = prior.detail_model

        base_model = base_model_class.estimate(
            mat_x, vec_y, space=space, rng=rng, prior=base_prior)
        base_prediction = base_model.predict_a(mat_x, return_std=False)

        detail_model = detail_model_class.estimate(
            mat_x, vec_y - base_prediction,
            space=space, rng=rng, prior=detail_prior)

        return cls(
            base_model,
            detail_model,
            space=space)

    def predict_transformed_a(
        self, mat_x_transformed: np.array, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        mat_x_transformed = coerce_array(mat_x_transformed)

        base_model = self.base_model
        detail_model = self.detail_model

        base_ys, base_std = base_model.predict_transformed_a(
            mat_x_transformed, return_std=return_std)
        detail_ys, detail_std = detail_model.predict_transformed_a(
            mat_x_transformed, return_std=return_std)

        vec_y = base_ys + detail_ys
        vec_std = None

        if return_std:
            assert base_std is not None
            assert detail_std is not None
            # sigma(X + Y) = sqrt(var(X) + var(Y) + cov(X, Y)
            # Assume that covariance is zero...
            vec_std = np.sqrt(base_std**2 + detail_std**2)

        return vec_y, vec_std

    def length_scales(self) -> np.ndarray:
        return np.ndarray([
            min(base_scale, detail_scale)
            for base_scale, detail_scale in zip(
                self.base_model.length_scales(),
                self.detail_model.length_scales())
        ])

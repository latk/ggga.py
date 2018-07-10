from .surrogate_model import SurrogateModel
from .space import Space
import typing as t
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore


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

    @classmethod
    def estimate(
        cls, xs: np.ndarray, ys: np.ndarray, *,
        space: Space,
        rng: RandomState,
        base_model_class: t.Type[SurrogateModel] = None,
        detail_model_class: t.Type[SurrogateModel] = None,
    ) -> 'SurrogateModelHierarchical':
        assert base_model_class is not None
        assert detail_model_class is not None

        base_model = base_model_class.estimate(
            xs, ys, space=space, rng=rng)
        base_prediction = base_model.predict_a(xs, return_std=False)

        detail_model = detail_model_class.estimate(
            xs, ys - base_prediction, space=space, rng=rng)

        return cls(
            base_model,
            detail_model,
            space=space)

    def predict_transformed_a(
        self, X: t.Iterable, *,
        return_std: bool=True,
    ):
        xs = np.array(list(X))

        base_model = self.base_model
        detail_model = self.detail_model

        if return_std:
            base_ys, base_std = base_model.predict_transformed_a(
                xs, return_std=True)
            detail_ys, detail_std = detail_model.predict_transformed_a(
                xs, return_std=True)
            return base_ys + detail_ys, base_std + detail_std

        base_ys = base_model.predict_transformed_a(
            xs, return_std=False)
        detail_ys = detail_model.predict_transformed_a(
            xs, return_std=False)
        return base_ys + detail_ys
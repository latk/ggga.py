import typing as t

from sklearn import neighbors  # type: ignore
from numpy.random import RandomState  # type: ignore
import numpy as np  # type: ignore

from .space import Space
from .surrogate_model import SurrogateModel


def weights(distances: np.ndarray) -> np.ndarray:
    # squared exponential kernel
    max_distance = np.max(distances)
    max_length_scale = 0.1
    length_scale = min(max_distance / 3, max_length_scale)  # seems to work
    return np.exp(-0.5 * (distances/length_scale)**2)


class SurrogateModelKNN(SurrogateModel):
    def __init__(
        self,
        estimator: neighbors.KNeighborsRegressor,
        *,
        space: Space,
        n_neighbors: int,
        xs_transformed: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        self.estimator = estimator
        self.space = space
        self.n_neighbors = n_neighbors
        self.xs_transformed = xs_transformed
        self.ys = ys

    def to_jsonish(self):
        return dict(
            model_class=type(self).__name__,
            k=self.n_neighbors,
        )

    @classmethod
    def estimate(
        cls, xs: np.ndarray, ys: np.ndarray, *,
        space: Space,
        rng: RandomState,  # pylint: disable=unused-argument
        prior: 'SurrogateModel',
        n_neighbors: int = 10,
        p: int = 2,  # pylint: disable=invalid-name,
    ) -> 'SurrogateModelKNN':
        assert prior is None or isinstance(prior, SurrogateModelKNN)
        estimator = neighbors.KNeighborsRegressor(
            n_neighbors,
            weights=weights,
            algorithm='kd_tree',
            p=p,
        )

        xs_transformed = np.array([space.into_transformed(x) for x in xs])
        ys = np.array(ys)
        estimator.fit(xs_transformed, ys)

        return cls(
            estimator,
            space=space,
            n_neighbors=n_neighbors,
            xs_transformed=xs_transformed,
            ys=ys,
        )

    def predict_transformed_a(
        self, mat_x: t.Iterable, *,
        return_std: bool = True,
    ):
        if not isinstance(mat_x, (list, np.ndarray)):
            mat_x = list(mat_x)
        requested_xs: np.ndarray = np.array(mat_x)

        estimator = self.estimator
        train_ys = self.ys

        y_mean = estimator.predict(requested_xs)

        if return_std:
            distances, neighbor_ixs = estimator.kneighbors(requested_xs)
            neighbor_ys = train_ys[neighbor_ixs]
            neighbor_weights = weights(distances)

            # # std of the model, using LOO-CV
            # #
            # #   for each neighbour:
            # #       for each request:
            # #           the estimate leaving out that neighbor.
            # n_neighbors = self.n_neighbors
            # n_requests = len(requested_xs)
            # y_mean_estimates = np.zeros(
            #     (n_neighbors, n_requests), dtype=float)
            # for i in range(n_neighbors):
            #     # the neighbour indices, leaving out the i'th neighbour
            #     indices = np.arange(0, n_neighbors - 1, dtype=int)
            #     indices[i:] += 1

            #     # for each request: the estimate using remaining neighbors
            #     y_mean_estimates[i, :] = np.average(
            #         neighbor_ys[:, indices],
            #         axis=1,
            #         weights=neighbor_weights[:, indices])
            # y_std = np.std(y_mean_estimates, axis=0)

            # std of the data
            y_var = np.average(
                (y_mean - neighbor_ys.T)**2,
                axis=0,
                weights=neighbor_weights.T)
            # y_var = np.mean((y_mean - neighbor_ys.T)**2, axis=0)
            y_std = np.sqrt(y_var)

            return y_mean, y_std

        return y_mean

    def length_scales(self) -> np.ndarray:
        # allow "useless" super call because the super-method is abstract.
        # pylint: disable=useless-super-delegation
        return super().length_scales()

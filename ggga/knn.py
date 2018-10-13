import typing as t

from sklearn import neighbors  # type: ignore
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

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
    def estimate(  # pylint: disable=arguments-differ
        cls, mat_x: np.ndarray, vec_y: np.ndarray, *,
        space: Space,
        rng: RandomState,  # pylint: disable=unused-argument
        prior: t.Optional[SurrogateModel],
        n_neighbors: int = 10,
        metric_p: int = 2,
        **kwargs,
    ) -> 'SurrogateModelKNN':

        assert prior is None or isinstance(prior, SurrogateModelKNN)

        if kwargs:
            raise TypeError(f"Unknown arguments: {sorted(kwargs)}")

        estimator = neighbors.KNeighborsRegressor(
            n_neighbors,
            weights=weights,
            algorithm='kd_tree',
            p=metric_p,
        )

        mat_x_transformed = np.array([
            space.into_transformed(x) for x in mat_x
        ])
        vec_y = np.array(vec_y)
        estimator.fit(mat_x_transformed, vec_y)

        return cls(
            estimator,
            space=space,
            n_neighbors=n_neighbors,
            xs_transformed=mat_x_transformed,
            ys=vec_y,
        )

    def predict_transformed_a(
        self, mat_x_transformed: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        if not isinstance(mat_x_transformed, (list, np.ndarray)):
            mat_x_transformed = list(mat_x_transformed)
        requested_xs: np.ndarray = np.array(mat_x_transformed)

        n_queries = len(requested_xs)
        n_neighbors = self.n_neighbors
        estimator = self.estimator
        train_ys = self.ys

        y_mean = estimator.predict(requested_xs)
        assert np.shape(y_mean) == (n_queries,)

        y_std = None
        if return_std:
            distances, neighbor_ixs = estimator.kneighbors(requested_xs)
            assert distances.shape == (n_queries, n_neighbors)
            assert np.shape(neighbor_ixs) == (n_queries, n_neighbors)

            mat_y_neighbor = train_ys[neighbor_ixs]
            mat_weights_neighbor = weights(distances)
            assert np.shape(mat_weights_neighbor) == (n_queries, n_neighbors)

            y_std = _std_from_data(
                y_mean, mat_y_neighbor, mat_weights_neighbor)
            assert np.shape(y_std) == (n_queries,)

        return y_mean, y_std


def _std_from_loocv(
    mat_x: np.ndarray,
    mat_y_neighbor: np.ndarray,
    mat_weights_neighbor: np.ndarray,
) -> np.ndarray:
    r"""std of the model, using LOO-CV

    Parameters
    ----------
    mat_x : ndarray, shape (n_queries, n_features)
        Locations at which to estimate the std.
        One coordinate per row.
    mat_y_neighbor : ndarray, shape (n_queries, n_neighbors)
        Values at the neighbors of the mat_x location.
        One neighbor per row.
        The i-th column contains the neighbors
        for the i-th coordinate in mat_x.
    mat_weights_neighbor : ndarray, shape (n_queries, n_neighbors)
        Weights for the neighbors, using the same shape as mat_y_neighbor.

    Pseudocode:

      for each neighbour:
          for each request:
              the estimate leaving out that neighbor.
      return std of the estimates
    """

    n_neighbors, n_queries = np.shape(mat_y_neighbor)
    assert len(mat_x) == n_queries
    assert np.shape(mat_y_neighbor) == np.shape(mat_weights_neighbor)

    mat_y = np.zeros((n_neighbors, n_queries), dtype=float)

    for i in range(n_neighbors):
        # the neighbour indices, leaving out the i'th neighbour
        vec_indices = np.arange(0, n_neighbors - 1, dtype=int)
        vec_indices[i:] += 1

        # for each request: the estimate using remaining neighbors
        mat_y[i, :] = np.average(
            mat_y_neighbor[:, vec_indices],
            axis=1,
            weights=mat_weights_neighbor[:, vec_indices])

    vec_std = np.std(mat_y, axis=0)
    assert np.shape(vec_std) == (n_queries,)
    return vec_std


def _std_from_data(
    vec_y: np.ndarray,
    mat_y_neighbor: np.ndarray,
    mat_weights_neighbor: np.ndarray,
) -> np.ndarray:
    n_queries, n_neighbors = np.shape(mat_y_neighbor)
    assert np.shape(mat_weights_neighbor) == np.shape(mat_y_neighbor)

    vec_y_var = np.average(
        (vec_y.reshape(n_queries, 1) - mat_y_neighbor)**2,
        axis=1,
        weights=mat_weights_neighbor,
    )
    assert np.shape(vec_y_var) == (n_queries,), \
        f"{vec_y_var.shape} n_neighbors={n_neighbors} n_queries={n_queries}"
    return np.sqrt(vec_y_var)

from sklearn.gaussian_process.kernels import (  # type: ignore
    Kernel, ConstantKernel, Matern, WhiteKernel, Sum, Product)
import scipy.linalg  # type: ignore
import warnings
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import typing as t
import attr
from sklearn.base import clone  # type: ignore

from .space import Space
from .util import fork_random_state, minimize_by_gradient
from .surrogate_model import SurrogateModel

# large parts of this code are “borrowed” from skopt (scikit-optimize),
# see https://github.com/scikit-optimize/scikit-optimize

TBounds = t.Tuple[float, float]


@attr.attrs(repr=False, frozen=True, cmp=False)
class SurrogateModelGPR(SurrogateModel):
    kernel: Kernel = attr.ib()
    X_train: np.ndarray = attr.ib()
    y_train: np.ndarray = attr.ib()
    alpha: np.ndarray = attr.ib()
    K_inv: np.ndarray = attr.ib()
    ys_mean: float = attr.ib()
    ys_min: float = attr.ib()
    lml: float = attr.ib()
    space: Space = attr.ib()

    def _all_config_items(self) -> t.Iterator[t.Tuple[str, t.Any]]:
        for key, value in self.kernel.get_params().items():
            keypart = key.split('__')[-1]
            if keypart[0] == 'k' and keypart[1:].isdigit():
                continue
            yield f"kernel_{key}", value

        yield 'ys_mean', self.ys_mean
        yield 'ys_min', self.ys_min
        yield 'lml', self.lml

    def __repr__(self):
        params_as_str = ''.join(
            f'\n    {key}={value}'
            for (key, value) in sorted(self._all_config_items()))
        return f'SurrogateModelGPR({self.kernel}{params_as_str})'

    def to_jsonish(self) -> dict:
        data: dict = dict()
        data['model_class'] = type(self).__name__
        data['kernel'] = dict()
        data['kernel']['symbolic'] = str(self.kernel)
        for k, v in self._all_config_items():
            if k.startswith('kernel_'):
                data['kernel'][k[len('kernel_'):]] = v
            else:
                data[k] = v
        return data

    def as_csv_row(self) -> list:
        return [value for key, value in sorted(self._all_config_items())]

    @classmethod
    def estimate(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        space: Space,
        rng: RandomState,
        prior: 'SurrogateModel',
        noise_bounds: TBounds = (1e-5, 1e5),
        amplitude_bounds: TBounds = (1e-5, 1e5),
        length_scale_bounds: t.Union[TBounds, t.List[TBounds]] = (1e-3, 1e3),
        n_restarts_optimizer: int = 2,
        matern_nu: float = 5/2,
    ) -> 'SurrogateModelGPR':
        n_dims = space.n_dims

        start_amplitude = 1.0
        assert start_amplitude in ClosedInterval(*amplitude_bounds)

        start_noise = 1.0
        assert start_noise in ClosedInterval(*noise_bounds)

        length_scale = np.ones(n_dims)
        if isinstance(length_scale_bounds, tuple):
            length_scale_bounds = [length_scale_bounds] * n_dims
        assert len(length_scale_bounds) == n_dims
        assert all(1.0 in ClosedInterval(*dim_bounds)
                   for dim_bounds in length_scale_bounds)

        ys_mean = np.mean(ys)
        ys_min = np.min(ys)

        if prior is not None:
            assert isinstance(prior, SurrogateModelGPR)
            kernel = clone(prior.kernel)
        else:

            amplitude = ConstantKernel(start_amplitude, amplitude_bounds)
            # TODO adjust length scale bounds
            kernel = Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=matern_nu)
            noise = WhiteKernel(start_noise, noise_bounds)
            kernel = Sum(Product(amplitude, kernel), noise)

        X_train = np.array([space.into_transformed(x) for x in xs])
        y_train = ys - ys_min
        relax_alpha = 1e-10
        lml = fit_kernel(
            kernel, X_train, y_train,
            rng=fork_random_state(rng),
            n_restarts_optimizer=n_restarts_optimizer,
            relax_alpha=relax_alpha)

        return cls.from_kernel(
            X_train, y_train, kernel,
            lml=lml, relax_alpha=relax_alpha,
            ys_min=ys_min, ys_mean=ys_mean,
            space=space,
        )

    @classmethod
    def from_kernel(
        cls, X_train: np.ndarray, y_train: np.ndarray, kernel: Kernel,
        lml: t.Optional[float],
        relax_alpha: float,
        ys_min: float,
        ys_mean: float,
        space: Space,
    ) -> 'SurrogateModelGPR':
        if lml is None:
            lml = -log_marginal_likelihood(
                kernel.theta, eval_gradient=False, kernel=kernel,
                X=X_train, y=y_train, relax_alpha=relax_alpha)

        # precompute matrices for prediction
        matrices_or_error = calculate_prediction_matrices(
            X_train, y_train,
            kernel=kernel,
            eval_gradient=False,
            relax_alpha=relax_alpha,
        )
        if isinstance(matrices_or_error, np.linalg.LinAlgError):
            exc = matrices_or_error
            exc.args = (
                "The kernel did not return a positive definite matrix. "
                "Please relax the alpha.", *exc.args)
            raise exc
        _, L, alpha, _ = matrices_or_error

        L_inv = scipy.linalg.solve_triangular(L.T, np.eye(L.shape[0]))
        K_inv = L_inv.dot(L_inv.T)

        return cls(
            kernel,
            alpha=alpha,
            K_inv=K_inv,
            X_train=X_train,
            y_train=y_train,
            ys_mean=ys_mean,
            ys_min=ys_min,
            lml=t.cast(float, lml),
            space=space)

    def predict_transformed_a(
        self, X: t.Iterable, *,
        return_std: bool=True,
    ):
        if not isinstance(X, (list, np.ndarray)):
            X = list(X)
        X = np.array(X)

        kernel = self.kernel
        alpha = self.alpha

        K_trans = kernel(X, self.X_train)
        y_mean = K_trans.dot(alpha)
        y_mean += self.ys_min  # undo normalization

        if return_std:
            # Compute variance of predictive distribution
            y_var = kernel.diag(X)
            y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, self.K_inv)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            y_std = np.sqrt(y_var)
            return y_mean, y_std

        return y_mean

    def __str__(self):
        return str(self.estimator.kernel_)

    def length_scales(self) -> np.ndarray:
        # assume the length scale is defined by the first kernel
        # that has a "length_scale" parameter and just take that.
        # This is the case for a Matern kernel.
        for key, value in sorted(self.kernel.get_params().items()):
            key = key.split('__')[-1]
            if key == 'length_scale':
                break
        else:
            return super().length_scales()

        if not len(np.shape(value)):
            value = [value]
        assert len(np.shape(value)) == 1
        return np.array(value)


@attr.s(frozen=True)
class ClosedInterval(object):
    lo: float = attr.ib()
    hi: float = attr.ib()

    def __contains__(self, x: float) -> bool:
        return self.lo <= x <= self.hi


def fit_kernel(
    kernel: Kernel,
    X: np.ndarray,
    y: np.ndarray, *,
    rng: RandomState,
    n_restarts_optimizer: int,
    relax_alpha: float = 0.0,
) -> float:
    """Assign kernel parameters with maximal log-marginal-likelihood.

    Parameters
    ----------
    kernel
        The prior kernel to tune.
        The current parameters (theta)
        are used as an optimization starting point.
        All parameter bounds must be finite.
        The kernel is modified in-place.
    X, y
        Supporting input observations, already transformed and normalized-
    rng
        Used to select additional optimizer start points.
    n_restarts_optimizer
        Number of additional starting points.
    relax_alpha
        Added to the covariance matrix diagonal
        to relax the matrix manipulation, should be unnecessary.
        Corresponds to adding a WhiteKernel.

    Returns
    -------
    The log marginal likelihood of the selected theta.
    """

    def obj_func(theta: np.ndarray, eval_gradient: bool = True):
        return log_marginal_likelihood(
            theta, eval_gradient=eval_gradient,
            kernel=kernel, X=X, y=y, relax_alpha=relax_alpha)

    bounds = kernel.bounds

    # start optimizing from prior kernel
    optimal_theta, optimal_lml = minimize_by_gradient(
        obj_func, kernel.theta, bounds=bounds)

    # add more restarts
    for _ in range(n_restarts_optimizer):
        theta_prior = rng.uniform(bounds[:, 0], bounds[:, 1])
        theta_posterior, lml = minimize_by_gradient(
            obj_func, theta_prior, bounds=bounds)
        if lml < optimal_lml:  # minimize the lml
            optimal_theta, optimal_lml = theta_posterior, lml

    # select result with minimal (negative) log-marginal likelihood
    kernel.theta = optimal_theta
    return -optimal_lml


def log_marginal_likelihood(
    theta: np.ndarray,
    eval_gradient: bool = False,
    *,
    kernel: Kernel,
    X: np.ndarray,
    y: np.ndarray,
    relax_alpha: float,
):
    """Calculate the (negated) log marginal likelihood for a specific theta.

    Parameters
    ----------
    theta
        The selected hyperparameters.
    eval_gradient
        Whether the gradient should be computed as well.
    kernel
        The kernel to which the theta should be applied.
    X, y
        Supporting input observations, already transformed and normalized.
    relax_alpha
        Noise level, relaxes the matrix manipulation problems.

    Returns
    -------
    float
        The log-marginal likelihood.
    ndarray
        The gradient, if requested.
    """
    kernel = kernel.clone_with_theta(theta)

    matrices_or_error = calculate_prediction_matrices(
        X, y,
        kernel=kernel,
        eval_gradient=eval_gradient,
        relax_alpha=relax_alpha,
    )
    if isinstance(matrices_or_error, np.linalg.LinAlgError):
        if eval_gradient:
            return -np.inf, np.zeros_like(theta)
        else:
            return -np.inf
    K, L, alpha, K_gradient = matrices_or_error

    # compute log-likelihood
    log_likelihood = -0.5 * y.dot(alpha)
    log_likelihood -= np.log(np.diag(L)).sum()
    log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)

    if eval_gradient:
        tmp = np.outer(alpha, alpha)
        tmp -= scipy.linalg.cho_solve((L, True), np.eye(K.shape[0]))
        # compute "0.5 * trace(tmp dot K_gradient"
        # without constructing the full matrix
        # as only the diagonal is required
        log_likelihood_gradient = 0.5 * np.einsum('ij,ijk->k', tmp, K_gradient)

        return -log_likelihood, -log_likelihood_gradient

    return -log_likelihood


def calculate_prediction_matrices(
    X: np.ndarray, y: np.ndarray, *,
    kernel: Kernel,
    eval_gradient: bool,
    relax_alpha: float,
) -> t.Union[
    np.linalg.LinAlgError,
    t.Tuple[np.ndarray, np.ndarray, np.ndarray, t.Optional[np.ndarray]],
]:
    K_gradient = None
    if eval_gradient:
        K, K_gradient = kernel(X, eval_gradient=True)
    else:
        K = kernel(X)

    K[np.diag_indices_from(K)] += relax_alpha

    try:
        L = scipy.linalg.cholesky(K, lower=True)
    except np.linalg.LinAlgError as exc:
        return exc

    # solve the system "K alpha = y" for alpha,
    # based on the cholesky factorization L.
    alpha = scipy.linalg.cho_solve((L, True), y)

    return K, L, alpha, K_gradient

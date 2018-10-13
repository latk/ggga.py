import warnings
import typing as t

from sklearn.gaussian_process.kernels import (  # type: ignore
    Kernel, ConstantKernel, Matern, WhiteKernel, Sum, Product)
from sklearn.base import clone  # type: ignore
from scipy.linalg import (  # type: ignore
    cholesky as cholesky_decomposition, cho_solve as cholesky_solve)
import scipy.linalg  # type: ignore
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import attr

from .space import Space
from .util import fork_random_state, minimize_by_gradient, coerce_array
from .surrogate_model import SurrogateModel

# large parts of this code are “borrowed” from skopt (scikit-optimize),
# see https://github.com/scikit-optimize/scikit-optimize

TBounds = t.Tuple[float, float]


@attr.attrs(repr=False, frozen=True, cmp=False)
class SurrogateModelGPR(SurrogateModel):
    kernel: Kernel = attr.ib()
    mat_x_train: np.ndarray = attr.ib()
    vec_y_train: np.ndarray = attr.ib()
    vec_alpha: np.ndarray = attr.ib()
    mat_k_inv: np.ndarray = attr.ib()
    y_expect: float = attr.ib()
    lml: float = attr.ib()
    space: Space = attr.ib()

    def _all_config_items(self) -> t.Iterator[t.Tuple[str, t.Any]]:
        for key, value in self.kernel.get_params().items():
            keypart = key.split('__')[-1]
            if keypart[0] == 'k' and keypart[1:].isdigit():
                continue
            yield f"kernel_{key}", value

        yield 'ys_expect', self.y_expect
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
        for key, value in self._all_config_items():
            if key.startswith('kernel_'):
                data['kernel'][key[len('kernel_'):]] = value
            else:
                data[key] = value
        return data

    def as_csv_row(self) -> list:
        return [value for key, value in sorted(self._all_config_items())]

    @classmethod
    def estimate(  # pylint: disable=arguments-differ,too-many-locals
        cls,
        mat_x: np.ndarray,
        vec_y: np.ndarray,
        *,
        space: Space,
        rng: RandomState,
        prior: t.Optional[SurrogateModel],
        noise_bounds: TBounds = (1e-5, 1e5),
        amplitude_bounds: TBounds = (1e-5, 1e5),
        length_scale_bounds: t.Union[TBounds, t.List[TBounds]] = (1e-3, 1e3),
        n_restarts_optimizer: int = 2,
        matern_nu: float = 5/2,
        **kwargs,
    ) -> 'SurrogateModelGPR':
        if kwargs:
            raise TypeError(f"Unknown arguments: {sorted(kwargs)}")

        kernel = _get_kernel_or_default(
            n_dims=space.n_dims,
            prior=prior,
            amplitude_bounds=amplitude_bounds,
            noise_bounds=noise_bounds,
            length_scale_bounds=length_scale_bounds,
            matern_nu=matern_nu,
        )

        y_expect = np.min(vec_y)

        mat_x_train = np.array([space.into_transformed(x) for x in mat_x])
        vec_y_train = vec_y - y_expect
        relax_alpha = 1e-10
        lml = fit_kernel(
            kernel, mat_x_train, vec_y_train,
            rng=fork_random_state(rng),
            n_restarts_optimizer=n_restarts_optimizer,
            relax_alpha=relax_alpha)

        return cls.from_kernel(
            mat_x_train, vec_y_train,
            kernel=kernel,
            lml=lml, relax_alpha=relax_alpha,
            y_expect=y_expect,
            space=space,
        )

    @classmethod
    def from_kernel(  # pylint: disable=too-many-locals
        cls, mat_x_train: np.ndarray, vec_y_train: np.ndarray, *,
        kernel: Kernel,
        lml: t.Optional[float],
        relax_alpha: float,
        y_expect: float,
        space: Space,
    ) -> 'SurrogateModelGPR':
        if lml is None:
            lml, _ = log_marginal_likelihood(
                kernel.theta, eval_gradient=False, kernel=kernel,
                mat_x=mat_x_train, vec_y=vec_y_train, relax_alpha=relax_alpha)
            lml = -lml
        assert lml is not None  # for type checker

        try:
            # precompute matrices for prediction
            mat_l, vec_alpha, _mat_k_gradient = \
                calculate_prediction_matrices(
                    mat_x_train, vec_y_train,
                    kernel=kernel,
                    eval_gradient=False,
                    relax_alpha=relax_alpha,
                )
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel did not return a positive definite matrix. "
                "Please relax the alpha.", *exc.args)
            raise exc

        mat_l_inv = scipy.linalg.solve_triangular(
            mat_l.T, np.eye(mat_l.shape[0]))
        mat_k_inv = mat_l_inv.dot(mat_l_inv.T)

        return cls(
            kernel,
            vec_alpha=vec_alpha,
            mat_k_inv=mat_k_inv,
            mat_x_train=mat_x_train,
            vec_y_train=vec_y_train,
            y_expect=y_expect,
            lml=lml,
            space=space)

    def predict_transformed_a(
        self, mat_x_transformed: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        mat_x_transformed = coerce_array(mat_x_transformed)

        kernel = self.kernel
        vec_alpha = self.vec_alpha

        mat_k_trans = kernel(mat_x_transformed, self.mat_x_train)
        vec_y = mat_k_trans.dot(vec_alpha)
        vec_y += self.y_expect  # undo normalization

        vec_y_std = None
        if return_std:
            # Compute variance of predictive distribution
            vec_y_var = kernel.diag(mat_x_transformed)
            vec_y_var -= np.einsum(
                "ki,kj,ij->k", mat_k_trans, mat_k_trans, self.mat_k_inv)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            vec_y_var_is_negative = vec_y_var < 0
            if np.any(vec_y_var_is_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                vec_y_var[vec_y_var_is_negative] = 0.0
            vec_y_std = np.sqrt(vec_y_var)

        return vec_y, vec_y_std

    def __str__(self):
        return str(self.estimator.kernel_)

    def length_scales(self) -> np.ndarray:
        # assume the length scale is defined by the first kernel
        # that has a "length_scale" parameter and just take that.
        # This is the case for a Matern kernel.
        for key, value in sorted(self.kernel.get_params().items()):
            key = key.split('__')[-1]

            if key == 'length_scale':
                if np.shape(value) == ():  # if there is a scalar scale
                    value = [value]
                assert len(np.shape(value)) == 1
                return np.array(value)

        return super().length_scales()


@attr.s(frozen=True)
class ClosedInterval:
    lo: float = attr.ib()
    hi: float = attr.ib()

    def __contains__(self, x: float) -> bool:
        return self.lo <= x <= self.hi


def fit_kernel(
    kernel: Kernel,
    mat_x: np.ndarray,
    vec_y: np.ndarray, *,
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
    mat_x, vec_y
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
        lml, lml_grad = log_marginal_likelihood(
            theta, eval_gradient=eval_gradient,
            kernel=kernel, mat_x=mat_x, vec_y=vec_y, relax_alpha=relax_alpha)
        if eval_gradient:
            return lml, lml_grad
        return lml

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
    mat_x: np.ndarray,
    vec_y: np.ndarray,
    relax_alpha: float,
) -> t.Tuple[float, t.Optional[np.ndarray]]:
    """Calculate the (negated) log marginal likelihood for a specific theta.

    Parameters
    ----------
    theta
        The selected hyperparameters.
    eval_gradient
        Whether the gradient should be computed as well.
    kernel
        The kernel to which the theta should be applied.
    mat_x, vec_y
        Supporting input observations, already transformed and normalized.
    relax_alpha
        Noise level, relaxes the matrix manipulation problems.

    Returns
    -------
    float
        The log-marginal likelihood.
    ndarray, optional
        The gradient, if requested.
    """
    kernel = kernel.clone_with_theta(theta)

    try:
        mat_l, vec_alpha, mat_k_gradient = \
            calculate_prediction_matrices(
                mat_x, vec_y,
                kernel=kernel,
                eval_gradient=eval_gradient,
                relax_alpha=relax_alpha,
            )
    except np.linalg.LinAlgError:
        vec_log_likelihood_gradient = None
        if eval_gradient:
            vec_log_likelihood_gradient = np.zeros_like(theta)
        return -np.inf, vec_log_likelihood_gradient

    # compute log-likelihood
    log_likelihood = -0.5 * vec_y.dot(vec_alpha)  # type: float
    log_likelihood -= np.log(np.diag(mat_l)).sum()
    log_likelihood -= len(mat_x) / 2 * np.log(2 * np.pi)

    if eval_gradient:
        mat_tmp = np.outer(vec_alpha, vec_alpha)
        mat_tmp -= cholesky_solve((mat_l, True), np.eye(len(mat_x)))
        # compute "0.5 * trace(tmp dot K_gradient)"
        # without constructing the full matrix
        # as only the diagonal is required
        vec_log_likelihood_gradient = \
            0.5 * np.einsum('ij,ijk->k', mat_tmp, mat_k_gradient)

        return -log_likelihood, -vec_log_likelihood_gradient

    return -log_likelihood, None


def calculate_prediction_matrices(
    mat_x: np.ndarray, vec_y: np.ndarray, *,
    kernel: Kernel,
    eval_gradient: bool,
    relax_alpha: float,
) -> t.Tuple[np.ndarray, np.ndarray, t.Optional[np.ndarray]]:
    r"""
    Arguments
    ---------
    mat_x : ndarray, shape (n, n_features)
        The observed samples.
    vec_y : ndarray, shape (n)
        The observation results.
    kernel : kernel function
    eval_gradient : bool
    relax_alpha : float
        Additional term to make the covariance matrix positive definite.

    Returns
    -------
    mat_l : ndarray, shape (n, n)
        The lower cholesky decomposition of the covariance matrix.
    vec_alpha : ndarray, shape (n)
        The observation weights. Solution to "mat_k vec_alpha = vec_y".
    mat_k_gradient : ndarray, shape (n, n), optional
        Covariance gradient. Only computed if *eval_gradient* is set.

    Raises
    ------
    np.linalg.LinAlgError
        When the Cholesky decomposition of mat_k fails,
        i.e. when the covariance matrix was not positive definite.
    """
    mat_k_gradient = None
    if eval_gradient:
        mat_k, mat_k_gradient = kernel(mat_x, eval_gradient=True)
    else:
        mat_k = kernel(mat_x)

    mat_k[np.diag_indices_from(mat_k)] += relax_alpha

    # may raise LinAlgError!
    # false positive: pylint: disable=unexpected-keyword-arg
    mat_l = cholesky_decomposition(mat_k, lower=True)

    # solve the system "K alpha = y" for alpha,
    # based on the cholesky factorization L.
    vec_alpha = cholesky_solve((mat_l, True), vec_y)

    return mat_l, vec_alpha, mat_k_gradient


def _get_kernel_or_default(
    *,
    n_dims: int,
    prior: t.Optional[Kernel],
    amplitude_bounds: TBounds,
    noise_bounds: TBounds,
    length_scale_bounds: t.Union[TBounds, t.List[TBounds]],
    matern_nu: float,
) -> Kernel:

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

    if prior is not None:
        assert isinstance(prior, SurrogateModelGPR)
        return clone(prior.kernel)

    amplitude = ConstantKernel(start_amplitude, amplitude_bounds)
    # TODO adjust length scale bounds
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        nu=matern_nu)
    noise = WhiteKernel(start_noise, noise_bounds)
    return Sum(Product(amplitude, kernel), noise)

r"""SurrogateModel based on Gaussian Process Regression (GPR)

The code is based on GPs as described in Rasmussen & Williams 2006,
in particular equations 2.23 and 2.24, and the algorithm 2.1::

    def predict(mat_x_train, vec_y_train, kernel, noise_level, mat_x):
        # note on notation: x = A \ b <=> A x = b
        mat_k = kernel(mat_x_train, mat_x_train)
        mat_k_trans = kernel(mat_x_train, mat_x)

        # Line 2:
        mat_l = cholesky(mat_k + noise_level mat_diag)
        # Line 3:
        vec_alpha = mat_l.T \ (mat_l \ vec_y)
        # Line 4:
        vec_y = mat_k_trans.T vec_alpha
        # Line 5:
        mat_v[:,k] = mat_l \ mat_k_trans[:,k]
        # Line 6:
        vec_y_var[k] = kernel(mat_x, mat_x)[k,k] - sum(mat_v[i,k] mat_v[i,k])
        # Line 7:
        lml = log p(vec_y_train | mat_x_train)
            = -½ vec_y_train.T vec_alpha - sum(log mat_l[i,i]) - n/2 log 2 pi
        # Line 8:
        return vec_y, vec_y_var, lml

Large parts of this code are based on skopt.learning.GaussianProcessRegressor
from skopt (scikit-optimize)
at https://github.com/scikit-optimize/scikit-optimize

"""

import warnings
import typing as t

from sklearn.gaussian_process.kernels import (  # type: ignore
        Kernel, ConstantKernel, Matern, Product)
from scipy.linalg import (  # type: ignore
    cholesky as cholesky_decomposition, cho_solve as cholesky_solve)
import scipy.linalg  # type: ignore
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import attr

from .space import Space
from .util import fork_random_state, minimize_by_gradient, coerce_array
from .surrogate_model import SurrogateModel

TBounds = t.Tuple[float, float]


@attr.attrs(repr=False, frozen=True, cmp=False)
class SurrogateModelGPR(SurrogateModel):
    kernel: Kernel = attr.ib()
    noise_level: float = attr.ib()
    noise_bounds: TBounds = attr.ib()
    mat_x_train: np.ndarray = attr.ib()
    vec_y_train: np.ndarray = attr.ib()
    vec_alpha: np.ndarray = attr.ib()
    mat_k_inv: np.ndarray = attr.ib()
    y_expect: float = attr.ib()
    y_amplitude: float = attr.ib()
    lml: float = attr.ib()
    space: Space = attr.ib()

    def _all_config_items(self) -> t.Iterator[t.Tuple[str, t.Any]]:
        for key, value in self.kernel.get_params().items():
            keypart = key.split('__')[-1]
            if keypart[0] == 'k' and keypart[1:].isdigit():
                continue
            yield f"kernel_{key}", value

        yield 'noise_level', self.noise_level
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

    @staticmethod
    def _normalized_from_ys(
        vec_y: np.ndarray,
    ) -> t.Tuple[np.ndarray, float, float]:
        r""" Transform ys to small positive numbers.

        Usually the GPR expectation is set to the mean of the observations.
        Here, we set it close to the minimum
        because we are more interested in finding a better minimum
        than finding an overall well-fitting model.

        Returns
        -------
        * vec_y (normalized)
        * y_amplitude
        * y_expect

        Example: inverse:

        >>> ys, y_amplitude, y_expect = SurrogateModelGPR._normalized_from_ys(
        ...     [1, 2, 3, 4])
        >>> SurrogateModelGPR._ys_from_normalized(
        ...     ys, y_amplitude=y_amplitude, y_expect=y_expect)
        array([1., 2., 3., 4.])

        Example: inverse with negative numbers:
        >>> ys, y_amplitude, y_expect = SurrogateModelGPR._normalized_from_ys(
        ...     [-5, 3, 18, -2])
        >>> SurrogateModelGPR._ys_from_normalized(
        ...     ys, y_amplitude=y_amplitude, y_expect=y_expect)
        array([-5.,  3., 18., -2.])
        """
        vec_y_train = np.array(vec_y, dtype=float)
        y_expect = np.min(vec_y)
        vec_y_train -= y_expect
        y_amplitude = np.mean(vec_y_train)
        if y_amplitude == 0:
            y_amplitude = 1.0
        vec_y_train /= y_amplitude
        vec_y_train += 0.05
        return vec_y_train, y_amplitude, y_expect

    @staticmethod
    def _ys_from_normalized(
        vec_y_normalized: np.ndarray, *, y_amplitude: float, y_expect: float,
    ) -> np.ndarray:
        vec_y = np.array(vec_y_normalized, dtype=float)
        vec_y -= 0.05
        vec_y *= y_amplitude
        vec_y += y_expect
        return vec_y

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
        amplitude_bounds: TBounds = None,
        length_scale_bounds: t.Union[TBounds, t.List[TBounds]] = (1e-3, 1e3),
        n_restarts_optimizer: int = 2,
        matern_nu: float = 5/2,
        **kwargs,
    ) -> 'SurrogateModelGPR':
        n_observations, n_features = np.shape(mat_x)
        assert np.shape(vec_y) == (n_observations,), repr(np.shape(vec_y))
        assert len(space.params) == n_features, repr(space.params)
        assert prior is None or isinstance(prior, SurrogateModelGPR)

        if kwargs:
            raise TypeError(f"Unknown arguments: {sorted(kwargs)}")

        vec_y_train, y_amplitude, y_expect = cls._normalized_from_ys(vec_y)
        mat_x_train = np.array([space.into_transformed(x) for x in mat_x])

        if amplitude_bounds is None:
            amplitude_hi = np.sum(vec_y_train**2)
            amplitude_lo = np.percentile(vec_y_train, 10)**2 * len(vec_y_train)
            assert amplitude_lo >= 0
            amplitude_bounds = (max(amplitude_lo, 2e-5) / 2, amplitude_hi * 2)

        amplitude_start = np.exp(np.mean(np.log(
            [amplitude_lo, amplitude_hi],
        )))

        kernel, noise_level, noise_bounds = _get_kernel_or_default(
            n_dims=space.n_dims,
            prior=prior,
            amplitude_start=amplitude_start,
            amplitude_bounds=amplitude_bounds,
            noise_bounds=noise_bounds,
            length_scale_bounds=length_scale_bounds,
            matern_nu=matern_nu,
        )

        noise_level, noise_bounds, vec_alpha, mat_k_inv, lml = \
            fit_kernel(
                kernel, mat_x_train, vec_y_train,
                rng=fork_random_state(rng),
                n_restarts_optimizer=n_restarts_optimizer,
                noise_level=noise_level,
                noise_bounds=noise_bounds,
            )

        return cls(
            kernel=kernel,
            noise_level=noise_level,
            noise_bounds=noise_bounds,
            mat_x_train=mat_x_train,
            vec_y_train=vec_y_train,
            vec_alpha=vec_alpha,
            mat_k_inv=mat_k_inv,
            y_expect=y_expect,
            y_amplitude=y_amplitude,
            lml=lml,
            space=space,
        )

    def predict_transformed_a(
        self, mat_x_transformed: np.ndarray, *,
        return_std: bool = True,
    ) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        mat_x_transformed = coerce_array(mat_x_transformed)

        kernel = self.kernel
        vec_alpha = self.vec_alpha

        mat_k_trans = kernel(mat_x_transformed, self.mat_x_train)
        vec_y = mat_k_trans.dot(vec_alpha)
        vec_y = self._ys_from_normalized(
            vec_y, y_amplitude=self.y_amplitude, y_expect=self.y_expect)

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
            vec_y_std = np.sqrt(vec_y_var) * self.y_amplitude

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
    kernel: Kernel, mat_x_train, vec_y_train, *,
    rng: RandomState,
    n_restarts_optimizer: int,
    noise_level: float,
    noise_bounds: TBounds,
) -> t.Tuple[float, TBounds, np.ndarray, np.ndarray, float]:
    # pylint: disable=too-many-locals,too-many-statements
    r"""Assign kernel parameters with maximal log-marginal-likelihood.

    Parameters
    ----------
    kernel
        The prior kernel to tune.
        The current parameters (theta)
        are used as an optimization starting point.
        All parameter bounds must be finite.
        The kernel is modified in-place.
    mat_x, vec_y
        Supporting input observations, already transformed and normalized.
    rng
        Used to select additional optimizer starting points.
    n_restarts_optimizer
        Number of additional starting points.
    noise_level, noise_bounds
        Noise hyperparameter prior and bounds.

    Returns
    -------
    noise_level, noise_level_bounds, vec_alpha, mat_k_inv, lml

    """
    n_observations = mat_x_train.shape[0]
    assert vec_y_train.shape == (n_observations,), repr(vec_y_train.shape)

    captured_lml = None
    captured_theta = None
    captured_noise_level = None
    captured_mat_l = None
    captured_vec_alpha = None

    def obj_func(theta: np.ndarray, eval_gradient: bool = True):
        r"""Calculate the negative log-marginal likelihood of a theta-vector.
        """
        # pylint: disable=invalid-unary-operand-type
        assert len(theta.shape) == 1, repr(theta.shape)

        kernel.theta = theta[:-1]
        noise_level = np.exp(theta[-1])

        mat_k_gradient = None
        if eval_gradient:
            mat_k, mat_k_gradient = kernel(mat_x_train, eval_gradient=True)
            mat_noise_gradient = \
                noise_level * np.eye(mat_x_train.shape[0])[:, :, np.newaxis]
            mat_k_gradient = np.dstack((mat_k_gradient, mat_noise_gradient))
        else:
            mat_k = kernel(mat_x_train)

        mat_k[np.diag_indices_from(mat_k)] += noise_level

        try:
            # false positive: pylint: disable=unexpected-keyword-arg
            mat_l = cholesky_decomposition(mat_k, lower=True)
        except np.linalg.LinAlgError:
            if eval_gradient:
                return np.inf, np.zeros_like(theta)
            return np.inf

        # solve the system "K alpha = y" for alpha,
        # based on the cholesky factorization L.
        vec_alpha = cholesky_solve((mat_l, True), vec_y_train)

        assert mat_k.shape == (n_observations, n_observations), \
            repr(mat_k.shape)
        assert vec_alpha.shape == (n_observations,), repr(vec_alpha.shape)

        lml: float = -0.5 * vec_y_train.dot(vec_alpha)
        lml -= np.log(np.diag(mat_l)).sum()
        lml -= len(mat_x_train) / 2 * np.log(2 * np.pi)

        vec_lml_grad: t.Optional[np.ndarray] = None
        if eval_gradient:
            mat_tmp = np.outer(vec_alpha, vec_alpha)
            mat_tmp -= cholesky_solve((mat_l, True), np.eye(len(mat_x_train)))
            # compute "0.5 * trace(tmp dot K_gradient)"
            # without constructing the full matrix
            # as only the diagonal is required
            vec_lml_grad = \
                0.5 * np.einsum('ij,ijk->k', mat_tmp, mat_k_gradient)

        # capture optimal theta incl. already computed matrices
        nonlocal captured_lml, captured_theta, captured_noise_level
        nonlocal captured_mat_l, captured_vec_alpha
        if captured_lml is None or lml > captured_lml:
            captured_lml = lml
            captured_theta = theta[:-1]
            captured_noise_level = noise_level
            captured_mat_l = mat_l
            captured_vec_alpha = vec_alpha

        # negate the lml for minimization
        if eval_gradient:
            assert vec_lml_grad is not None
            return -lml, -vec_lml_grad
        return -lml

    # Perform multiple minimization runs.
    # Usually these functions return the output,
    # but here we capture the results in the objective function.
    bounds = np.vstack((kernel.bounds, np.log([noise_bounds])))
    initial_theta = np.hstack((kernel.theta, np.log([noise_level])))
    minimize_by_gradient(obj_func, initial_theta, bounds=bounds)
    for _ in range(n_restarts_optimizer):
        theta_prior = rng.uniform(bounds[:, 0], bounds[:, 1])
        minimize_by_gradient(obj_func, theta_prior, bounds=bounds)

    assert captured_theta is not None
    assert captured_noise_level is not None
    assert captured_lml is not None
    assert captured_mat_l is not None
    assert captured_vec_alpha is not None

    kernel.theta = captured_theta
    noise_level = captured_noise_level

    # Precompute arrays needed at prediction
    mat_l, vec_alpha = captured_mat_l, captured_vec_alpha
    mat_l_inv = scipy.linalg.solve_triangular(mat_l.T, np.eye(mat_l.shape[0]))
    mat_k_inv = mat_l_inv.dot(mat_l_inv.T)

    return noise_level, noise_bounds, vec_alpha, mat_k_inv, captured_lml


def _get_kernel_or_default(
    *,
    n_dims: int,
    prior: t.Optional[SurrogateModelGPR],
    noise_bounds: TBounds,
    amplitude_start: float,
    amplitude_bounds: TBounds,
    length_scale_bounds: t.Union[TBounds, t.List[TBounds]],
    matern_nu: float,
) -> t.Tuple[Kernel, float, TBounds]:

    assert amplitude_start in ClosedInterval(*amplitude_bounds)

    start_noise = 1.0
    assert start_noise in ClosedInterval(*noise_bounds)

    length_scale = np.ones(n_dims)
    if isinstance(length_scale_bounds, tuple):
        length_scale_bounds = [length_scale_bounds] * n_dims
    assert len(length_scale_bounds) == n_dims
    assert all(1.0 in ClosedInterval(*dim_bounds)
               for dim_bounds in length_scale_bounds)

    amplitude = ConstantKernel(amplitude_start, amplitude_bounds)
    # TODO adjust length scale bounds
    main_kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        nu=matern_nu)
    kernel = Product(amplitude, main_kernel)
    noise_level = start_noise

    if prior is not None:
        assert isinstance(prior, SurrogateModelGPR)
        assert sorted(kernel.get_params().keys()) \
            == sorted(prior.kernel.get_params().keys())
        kernel.theta = prior.kernel.theta
        noise_level = prior.noise_level

    return kernel, noise_level, noise_bounds

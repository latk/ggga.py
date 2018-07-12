from sklearn.gaussian_process.gpr import (  # type: ignore
    GaussianProcessRegressor)
from sklearn.gaussian_process.kernels import (  # type: ignore
    ConstantKernel, Matern, WhiteKernel, Sum, Product)
from scipy.linalg import solve_triangular  # type: ignore
import warnings
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import typing as t

from .space import Space
from .util import fork_random_state
from .surrogate_model import SurrogateModel

# large parts of this code are “borrowed” from skopt (scikit-optimize),
# see https://github.com/scikit-optimize/scikit-optimize


class SurrogateModelGPR(SurrogateModel):
    def __init__(
        self,
        estimator: GaussianProcessRegressor,
        *,
        ys_mean: float,
        ys_min: float,
        space: Space,
    ) -> None:
        self.estimator = estimator
        self.ys_mean = ys_mean
        self.ys_min = ys_min
        self.space = space

    def __repr__(self):
        # return f'SurrogateModelGPR({self.estimator.kernel_})'
        params = self.estimator.kernel_.get_params()
        params_as_str = ''.join(f'\n    {key}={value}'
                                for (key, value) in sorted(params.items()))
        return f'SurrogateModelGPR({params_as_str})'

    @classmethod
    def estimate(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        space: Space,
        rng: RandomState,
        prior: 'SurrogateModel',
        noise_bounds: t.Tuple[float, float] = (1e-3, 1e2)
        n_restarts_optimizer: int = 10
    ) -> 'SurrogateModelGPR':
        n_dims = space.n_dims

        if prior is not None:
            assert isinstance(prior, SurrogateModelGPR)
            prior_kernel = prior.estimator.kernel_
        else:
            # TODO adjust amplitude bounds
            amplitude = ConstantKernel(1.0, (1e-2, 1e3))
            # TODO adjust length scale bounds
            kernel = Matern(
                length_scale=np.ones(n_dims),
                length_scale_bounds=[(1e-3, 1e3)] * n_dims,
                nu=5/2)
            noise = WhiteKernel(1.0, noise_bounds)
            prior_kernel = Sum(Product(amplitude, kernel), noise)

        estimator = GaussianProcessRegressor(
            kernel=prior_kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            copy_X_train=True,
            random_state=fork_random_state(rng),
        )

        estimator.fit([space.into_transformed(x) for x in xs], ys)

        # find the WhiteKernel params and turn it off for prediction

        def param_for_white_kernel_in_sum(kernel, kernel_str=""):
            if kernel_str:
                kernel_str += '__'
            if isinstance(kernel, Sum):
                for param, child in kernel.get_params(deep=False).items():
                    if isinstance(child, WhiteKernel):
                        return kernel_str + param
                    child_str = param_for_white_kernel_in_sum(
                        child, kernel_str + param)
                    if child_str is not None:
                        return child_str
            return None

        # white_kernel_param = param_for_white_kernel_in_sum(estimator.kernel_)
        # if white_kernel_param is not None:
        #     estimator.kernel_.set_params(**{
        #         white_kernel_param: WhiteKernel(noise_level=0.0)})

        # Precompute arrays needed at prediction
        L_inv = solve_triangular(estimator.L_.T, np.eye(estimator.L_.shape[0]))
        estimator.K_inv_ = L_inv.dot(L_inv.T)

        estimator.y_train_mean_ = estimator._y_train_mean

        return cls(
            estimator,
            ys_mean=np.mean(ys),
            ys_min=np.min(ys),
            space=space)

    def predict_transformed_a(
        self, X: t.Iterable, *,
        return_std: bool=True,
    ):
        if not isinstance(X, (list, np.ndarray)):
            X = list(X)
        X = np.array(X)

        estimator = self.estimator
        kernel = estimator.kernel_
        alpha = estimator.alpha_

        K_trans = kernel(X, estimator.X_train_)
        y_mean = K_trans.dot(alpha)
        y_mean = estimator.y_train_mean_ + y_mean  # undo normalization

        if return_std:
            K_inv = estimator.K_inv_

            # Compute variance of predictive distribution
            y_var = kernel.diag(X)
            y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

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

# Gaussian Process Guided Genetic Algorithm

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
from sklearn.gaussian_process.gpr import (  # type: ignore
    GaussianProcessRegressor)
from sklearn.gaussian_process.kernels import (  # type: ignore
    ConstantKernel, Matern, WhiteKernel, Sum, Product)
import scipy.stats  # type: ignore
import typing as t
import asyncio
from .space import Param, Real, Integer, Space  # noqa: F401 (public reexport)
from scipy.linalg import solve_triangular, cho_solve  # type: ignore
import warnings

# large parts of this code are “borrowed” from skopt (scikit-optimize),
# see https://github.com/scikit-optimize/scikit-optimize


def _fork_random_state(rng):
    return RandomState(rng.randint(2**32 - 1))


class SurrogateModel(object):
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

    @classmethod
    def estimate(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        space: Space,
        rng: RandomState,
    ) -> 'SurrogateModel':
        n_dims = space.n_dims

        # TODO adjust amplitude bounds
        amplitude = ConstantKernel(1.0, (1e-2, 1e3))
        # TODO adjust length scale bounds
        kernel = Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(1e-3, 1e3)] * n_dims,
            nu=5/2)
        noise = WhiteKernel(1.0, (1e-3, 1e4))
        estimator = GaussianProcessRegressor(
            kernel=Sum(Product(amplitude, kernel), noise),
            normalize_y=True,
            n_restarts_optimizer=2,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            copy_X_train=True,
            random_state=_fork_random_state(rng),
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

    def predict(self, sample: list):
        mean, std = self.predict_a([sample])
        return mean[0], std[0]

    def predict_a(self, multiple_samples: list):
        mean, std = self.predict_transformed_a(
            self.space.into_transformed(sample)
            for sample in multiple_samples)
        return mean, std

    def predict_transformed_a(
        self, X: t.Iterable, *,
        return_std: bool=True,
        return_cov: bool=False,
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

        if return_cov:
            assert not return_std
            v = cho_solve((estimator.L_, True), K_trans.T)
            y_cov = kernel(X) - K_trans.dot(v)
            return y_mean, y_cov

        elif return_std:
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


class Logger(object):
    def record_evaluations(self, individuals, *, space):
        print("[INFO] evaluations:")
        param_names = [p.name for p in space.params]
        param_formats = \
            ['{:.5f}' if isinstance(p, Real) else '{}' for p in space.params]
        print(tabularize(
            header=['utility', *param_names],
            formats=['{:.2f}', *param_formats],
            data=[[ind.fitness, *ind.sample] for ind in individuals],
        ))

    def announce_new_generation(
        self, gen: int, *,
        model: SurrogateModel,
        relscale: float,
    ) -> None:
        print("[INFO] starting generation #{}".format(gen))
        print("       relscale {:.5f}".format(relscale))
        print("       estimator: {!r}".format(model.estimator))


class Individual(object):
    def __init__(self, sample, fitness):
        self.sample = sample
        self.fitness = fitness


def expected_improvement(mean, std, fmin):
    norm = scipy.stats.norm
    z = -(mean - fmin) / std
    ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z)
    return -ei


async def minimize(
    objective: t.Callable[[list, RandomState], t.Awaitable[float]],
    *,
    space: Space,
    popsize: int=10,
    max_nevals: int=100,
    logger: Logger=None,
    rng: RandomState,
    relscale_initial=0.3,
    relscale_attenuation=0.9,
):

    if logger is None:
        logger = Logger()

    assert popsize < max_nevals

    population = [
        Individual(space.sample(rng=rng), None)
        for _ in range(popsize)
    ]

    all_evaluations = []
    all_models = []

    population_fitness = await asyncio.gather(*(
        objective(ind.sample, _fork_random_state(rng))
        for ind in population))
    for ind, fitness in zip(population, population_fitness):
        ind.fitness = fitness
        all_evaluations.append(ind)
    logger.record_evaluations(population, space=space)

    model = SurrogateModel.estimate(
        [ind.sample for ind in all_evaluations],
        [ind.fitness for ind in all_evaluations],
        space=space, rng=rng)
    all_models.append(model)

    def suggest_next_candidate(
        parent_sample, *, relscale, fmin, rng, breadth, model,
    ):
        candidate_samples = [
            space.mutate(parent_sample, relscale=relscale, rng=rng)
            for _ in range(breadth)]
        candidate_mean, candidate_std = model.predict_a(candidate_samples)
        candidate_ei = expected_improvement(
            candidate_mean, candidate_std, fmin)
        return candidate_samples[np.argmax(candidate_ei)]

    generation = 0
    while len(all_evaluations) < max_nevals:
        generation += 1
        relscale = relscale_initial * (relscale_attenuation**(generation - 1))

        logger.announce_new_generation(
            generation, model=model, relscale=relscale)

        # generate new individuals
        offspring: t.List[Individual] = []
        for parent in population:
            candidate_sample = parent.sample
            candidate_chain_length = 5
            for i in range(candidate_chain_length):
                candidate_sample = suggest_next_candidate(
                    candidate_sample,
                    relscale=relscale * (relscale_attenuation ** i),
                    fmin=parent.fitness,
                    rng=rng,
                    breadth=20,
                    model=model,
                )
            offspring.append(Individual(candidate_sample, None))

        # evaluate new individuals
        offspring_fitness = await asyncio.gather(*(
            objective(ind.sample, _fork_random_state(rng))
            for ind in offspring))
        for ind, fitness in zip(offspring, offspring_fitness):
            ind.fitness = fitness
            all_evaluations.append(ind)
        logger.record_evaluations(offspring, space=space)

        # fit next model
        model = SurrogateModel.estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=space, rng=rng)
        all_models.append(model)

        # select new population
        selected = []
        rejected = []
        for i in range(popsize):
            a, b = offspring[i], population[i]
            if b.fitness < a.fitness:
                a, b = b, a
            selected.append(a)
            rejected.append(b)

        # replace worst selected elements
        replace_worst_n = popsize - 3
        replacement_locations = \
            np.argsort([ind.fitness for ind in rejected])[:replace_worst_n]
        selected.extend(rejected[i] for i in replacement_locations)
        selected = sorted(selected, key=lambda ind: ind.fitness)[:popsize]

        population = selected

    best_individual = \
        population[np.argmin([ind.fitness for ind in population])]
    return best_individual.sample, best_individual.fitness, OptimizationResult(
        all_evaluations=all_evaluations,
        best_individual=best_individual,
        all_models=all_models,
    )


class OptimizationResult(object):
    def __init__(
        self, *,
        all_evaluations,
        best_individual,
        all_models,
    ) -> None:
        self.all_individuals = all_evaluations
        self.best_individual = best_individual
        self.all_models = all_models


def tabularize(
        header: t.List[str],
        formats: t.List[str],
        data: t.Iterable[list],
) -> str:
    columns = [[str(h)] for h in header]
    for row in data:
        for col, f, d in zip(columns, formats, row):
            col.append(f.format(d))
    col_size = [max(len(d) for d in col) for col in columns]
    out = []
    out.append(' '.join('-' * size for size in col_size))
    for i in range(len(columns[0])):
        out.append(' '.join(
            col[i].rjust(size) for col, size in zip(columns, col_size)))
    out[0], out[1] = out[1], out[0]
    return '\n'.join(out)

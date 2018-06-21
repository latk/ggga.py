# Gaussian Process Guided Genetic Algorithm

import sys
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import abc
from skopt.learning import GaussianProcessRegressor  # type: ignore
from skopt.learning.gaussian_process.kernels import (  # type: ignore
        ConstantKernel, Matern, WhiteKernel)
import scipy.stats  # type: ignore
import typing as t

T = t.TypeVar('T')


class Param(abc.ABC, t.Generic[T]):
    def __init__(self, name, flag):
        self.name = name
        self.flag = flag

    @abc.abstractmethod
    def sample(self, *, rng: RandomState, lo=None, hi=None) -> T:
        pass

    @abc.abstractmethod
    def mutate(self, value: T, *, rng: RandomState, relscale: float) -> T:
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractmethod
    def is_valid(self, value: T) -> bool:
        pass

    @abc.abstractmethod
    def is_valid_transformed(self, value) -> bool:
        pass

    @abc.abstractmethod
    def into_transformed(self, value: T):
        pass

    @abc.abstractmethod
    def from_transformed(self, value) -> T:
        pass

    def into_transformed_a(self, values: list) -> list:
        return [self.into_transformed(x) for x in values]

    def from_transformed_a(self, values: list) -> list:
        return [self.from_transformed(x) for x in values]

    @abc.abstractmethod
    def transformed_bounds(self) -> tuple:
        pass


class Integer(Param[int]):
    lo: int
    hi: int

    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(
        self, *,
        rng: RandomState,
        lo: int=None,
        hi: int=None,
    ) -> int:
        if lo is None:
            lo = self.lo
        else:
            assert self.lo <= lo
        if hi is None:
            hi = self.hi
        else:
            assert hi <= self.hi
        return rng.randint(lo, hi + 1)

    def mutate(self, value, *, rng: RandomState, relscale: float) -> int:
        retries = 20
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
            relscale *= 0.8
        raise RuntimeError("mutation failed to produce values within bounds")

    @property
    def size(self) -> int:
        return self.hi - self.lo

    def is_valid(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def is_valid_transformed(self, value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: int) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> int:
        return int(np.round(value * self.size + self.lo))

    def transformed_bounds(self) -> t.Tuple[float, float]:
        return (0.0, 1.0)


class Real(Param[float]):
    lo: float
    hi: float

    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(
        self, *,
        rng: RandomState,
        lo: float=None,
        hi: float=None,
    ) -> float:
        if lo is None:
            lo = self.lo
        else:
            assert self.lo <= lo
        if hi is None:
            hi = self.hi
        else:
            assert hi <= self.hi
        size = hi - lo
        return rng.random_sample() * size + lo

    def mutate(self, value, *, rng: RandomState, relscale: float) -> float:
        retries = 20
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
            relscale *= 0.8
        raise RuntimeError("mutation failed to produce values within bounds")

    @property
    def size(self) -> float:
        return self.hi - self.lo

    def is_valid(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def is_valid_transformed(self, value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: float) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> float:
        return value * self.size + self.lo

    def transformed_bounds(self) -> t.Tuple[float, float]:
        return (0.0, 1.0)


class Space(object):
    def __init__(
        self, *params: Param,
        constraints: t.List[t.Callable[[list], bool]]=None,
        constrained_bounds_suggestions: t.List[t.Callable[[list], dict]]=None,
    ) -> None:
        if constraints is None:
            constraints = []
        if constrained_bounds_suggestions is None:
            constrained_bounds_suggestions = []

        assert all(isinstance(p, Param) for p in params)
        assert all(callable(c) for c in constraints)
        assert all(callable(s) for s in constrained_bounds_suggestions)

        self.params = params
        self.constraints = constraints
        self.constrained_bounds_suggestions = constrained_bounds_suggestions

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self, *, rng: RandomState) -> list:
        retries = 10
        bounds: t.Dict[str, tuple] = dict()

        def merge_lo_hi(llo, lhi, rlo, rhi):
            if   llo is None:   lo = rlo            # noqa
            elif rlo is None:   lo = llo            # noqa
            else:               lo = max(llo, rlo)  # noqa

            if   lhi is None:   hi = rhi            # noqa
            elif rhi is None:   hi = lhi            # noqa
            else:               hi = min(lhi, rhi)  # noqa

            if lo is not None and hi is not None:
                assert lo <= hi
            return lo, hi

        for _ in range(retries):
            s = []
            for param in self.params:
                lo, hi = bounds.get(param.name, (None, None))
                s.append(param.sample(rng=rng, lo=lo, hi=hi))
            if all(c(s) for c in self.constraints):
                return s
            for suggestion in self.constrained_bounds_suggestions:
                for k, v in suggestion(s).items():
                    if v is None:
                        continue
                    llo, lhi = bounds.get(k, (None, None))
                    rlo, rhi = v
                    bounds[k] = merge_lo_hi(llo, lhi, rlo, rhi)

        raise RuntimeError("Could not find valid sample")

    def mutate(self, sample: list, *,
               rng: RandomState, relscale: float) -> list:
        return [
            p.mutate(x, rng=rng, relscale=relscale)
            for p, x in zip(self.params, sample)
        ]

    def is_valid(self, sample) -> bool:
        return all(p.is_valid(v) for p, v in zip(self.params, sample)) \
                and all(c(sample) for c in self.constraints)

    def into_transformed(self, sample: list) -> list:
        return [p.into_transformed(v) for p, v in zip(self.params, sample)]

    def from_transformed(self, sample: list) -> list:
        return [p.from_transformed(v) for p, v in zip(self.params, sample)]


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
            kernel=amplitude * kernel + noise,
            normalize_y=True,
            noise=0,
            n_restarts_optimizer=2,
            random_state=RandomState(rng.randint(2**32 - 1)),
        )

        estimator.fit([space.into_transformed(x) for x in xs], ys)

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

    def predict_transformed_a(self, multiple_transformed_samples: t.Iterable):
        if not isinstance(multiple_transformed_samples, (list, np.ndarray)):
            multiple_transformed_samples = list(multiple_transformed_samples)
        return self.estimator.predict(
            multiple_transformed_samples, return_std=True)


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


def minimize(
    objective: t.Callable[[list, RandomState], float],
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

    for ind in population:
        ind.fitness = objective(ind.sample, rng)
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
        for ind in offspring:
            ind.fitness = objective(ind.sample, rng)
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

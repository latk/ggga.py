# Gaussian Process Guided Genetic Algorithm

import sys
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import abc
from skopt.learning import GaussianProcessRegressor  # type: ignore
from skopt.learning.gaussian_process.kernels import (  # type: ignore
        ConstantKernel, Matern, WhiteKernel)
import typing as t

T = t.TypeVar('T')


class Param(abc.ABC, t.Generic[T]):
    def __init__(self, name, flag):
        self.name = name
        self.flag = flag

    @abc.abstractmethod
    def sample(self, *, rng: RandomState) -> T:
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


class Integer(Param[int]):
    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(self, *, rng: RandomState) -> int:
        return rng.randint(self.lo, self.hi + 1)

    def mutate(self, value, *, rng: RandomState, relscale: float) -> int:
        retries = 10
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
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


class Real(Param[float]):
    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(self, *, rng: RandomState) -> float:
        return rng.random_sample() * self.size + self.lo

    def mutate(self, value, *, rng: RandomState, relscale: float) -> float:
        retries = 10
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
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


class Space(object):
    def __init__(self, *params: Param, constraints=None) -> None:
        if constraints is None:
            constraints = []

        assert all(isinstance(p, Param) for p in params)
        assert all(callable(c) for c in constraints)

        self.params = params
        self.constraints = constraints

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self, *, rng: RandomState) -> list:
        retries = 10
        for _ in range(retries):
            s = [p.sample(rng=rng) for p in self.params]
            if all(c(s) for c in self.constraints):
                return s
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
        mean, std = self.estimator.predict(
            [self.space.into_transformed(sample)
                for sample in multiple_samples],
            return_std=True)
        return mean, std


class Logger(object):
    def log_evaluation(self, x, y) -> None:
        print("[INFO] evaluated {} to {}".format(x, y))

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


def minimize(
        objective: t.Callable[[list, RandomState], float],
        *,
        space: Space,
        popsize: int=10,
        max_nevals: int=100,
        logger: Logger=None,
        rng: RandomState):

    if logger is None:
        logger = Logger()

    assert popsize < max_nevals

    population = [
        Individual(space.sample(rng=rng), None)
        for _ in range(popsize)
    ]

    all_evaluations = []

    for ind in population:
        ind.fitness = objective(ind.sample, rng)
        all_evaluations.append(ind)
        logger.log_evaluation(ind.sample, ind.fitness)

    model = SurrogateModel.estimate(
        [ind.sample for ind in all_evaluations],
        [ind.fitness for ind in all_evaluations],
        space=space, rng=rng)

    generation = 0
    while len(all_evaluations) < max_nevals:
        generation += 1
        relscale = 0.1 * (0.9 ** (generation - 1))

        logger.announce_new_generation(
            generation, model=model, relscale=relscale)

        # generate new individuals
        new: t.List[Individual] = []
        retries = 5
        for parent in population:
            for _ in range(retries):
                child = Individual(
                    space.mutate(parent.sample, relscale=relscale, rng=rng),
                    None,
                )
                # TODO check for EI instead of blind minimization
                child_prediction, parent_prediction = \
                    model.predict_a([child.sample, parent.sample])[0]
                if child_prediction < parent_prediction:
                    new.append(child)
                    break

        # evaluate new individuals
        for ind in new:
            ind.fitness = objective(ind.sample, rng)
            all_evaluations.append(ind)
            logger.log_evaluation(ind.sample, ind.fitness)
        population.extend(new)

        # select new population
        model = SurrogateModel.estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=space, rng=rng)
        pop_predictions, pop_predictions_std = \
            model.predict_a([ind.sample for ind in population])
        selection_index = np.argsort(pop_predictions)[:popsize]
        population = [population[i] for i in selection_index]

    return population[0].sample, population[0].fitness

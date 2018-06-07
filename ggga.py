# Gaussian Process Guided Genetic Algorithm

import numpy as np
from numpy.random import RandomState
import abc
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


class Param(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def sample(self, *, random: RandomState):
        pass

    @abc.abstractmethod
    def mutate(self, value, *, random: RandomState):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractmethod
    def is_valid(self, value) -> bool:
        pass

    @abc.abstractmethod
    def is_valid_transformed(self, value) -> bool:
        pass

    @abc.abstractmethod
    def into_transformed(self, value):
        pass

    @abc.abstractmethod
    def from_transformed(self, value):
        pass


class Integer(Param):
    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(self, *, random: RandomState) -> int:
        return random.randint(self.lo, self.hi + 1)

    def mutate(self, value, *, random: RandomState, relscale: float) -> int:
        assert relscale * 2 < self.size  # not strictly necessary
        retries = 10
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + random.standard_normal() * relscale
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


class Real(Param):
    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(self, *, random: RandomState) -> float:
        return random.random_sample() * self.size + self.lo

    def mutate(self, value, *, random: RandomState, relscale: float) -> float:
        assert relscale * 2 < self.size  # not strictly necessary
        retries = 10
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + random.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
        raise RuntimeError("mutation failed to produce values within bounds")

    @property
    def size(self) -> float:
        return self.hi - self.lo

    def is_valid(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def into_transformed(self, value: float) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> float:
        return value * self.size + self.lo


class Space(object):
    def __init__(self, *params: Param, constraints=None):
        if constraints is None:
            constraints = []

        assert all(isinstance(p, Param) for p in params)
        assert all(callable(c) for c in constraints)

        self.params = params
        self.constraints = constraints

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self) -> list:
        retries = 5
        for _ in range(retries):
            s = [p.sample() for p in self.params]
            if all(c(s) for c in self.constraints):
                return s
        raise RuntimeError("Could not find valid sample")

    def mutate(self, sample: list, *,
               random: RandomState, relscale: float) -> list:
        return [
            p.mutate(x, random=random, relscale=relscale)
            for p, x in zip(self.params, sample)
        ]

    def is_valid(self, sample) -> bool:
        return all(p.is_valid(v) for p, v in zip(self.params, sample)) \
                and all(c(sample) for c in self.constraints)

    def into_transformed(self, sample) -> list:
        return [p.into_transformed(v) for p, v in zip(self.params, sample)]

    def from_transformed(self, sample) -> list:
        return [p.from_transformed(v) for p, v in zip(self.params, sample)]


class SurrogateModel(object):
    def __init__(
        self,
        estimator: GaussianProcessRegressor,
        *,
        ys_mean: float,
        ys_min: float,
        space: Space,
    ):
        self.estimator = estimator
        self.ys_mean = ys_mean
        self.ys_min = ys_min
        self.space = space

    @classmethod
    def estimate(
        cls,
        xs: np.ndarray,
        ys: np.ndarray,
        space: Space
    ) -> 'SurrogateModel':
        n_dims = space.n_dims

        # TODO adjust amplitude bounds
        amplitude = ConstantKernel(1.0, (1e-2, 1e3))
        # TODO adjust length scale bounds
        kernel = Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(1e-3, 1e3)] * n_dims,
            nu=5/2)
        estimator = GaussianProcessRegressor(
                kernel=amplitude * kernel,
                normalize_y=True,
                noise='gaussian',
                n_restarts_optimizer=2)

        estimator.fit([space.into_transformed(x) for x in xs], ys)

        return cls(
            estimator,
            ys_mean=np.mean(ys),
            ys_min=np.min(ys),
            space=space)

    def predict(self, x: list):
        mean, std = self.predict_a([x])
        return mean[0], std[0]

    def predict_a(self, xs: list):
        mean, std = self.estimator.predict(
            [self.space.into_transformed(x) for x in xs],
            return_std=True)
        return mean, std


class Logger(object):
    def log_evaluation(self, x, y):
        print("[INFO] evaluated {} to {}".format(x, y))


class Individual(object):
    def __init__(self, sample, fitness):
        self.sample = sample
        self.fitness = fitness


def minimize(
        objective: 'Callable[list] -> float',
        *,
        space: Space,
        popsize: int=10,
        max_nevals: int=100,
        logger: Logger=None):

    if logger is None:
        logger = Logger()

    assert popsize < max_nevals

    population = [Individual(space.sample(), None) for _ in range(popsize)]

    all_evaluations = []

    for ind in population:
        ind.fitness = objective(ind.sample)
        all_evaluations.append(ind)
        logger.log_evaluation(ind.sample, ind.fitness)

    model = SurrogateModel.estimate(
        [ind.sample for ind in all_evaluations],
        [ind.fitness for ind in all_evaluations],
        space)

    generation = 0
    while len(all_evaluations) < max_nevals:
        generation += 1

        # generate new individuals
        new = []
        retries = 5
        for parent in population:
            for _ in range(retries):
                child = Individual(space.mutate(parent.sample), None)
                # TODO check for EI instead of blind minimization
                child_prediction, parent_prediction = \
                    model.predict_a([child.sample, parent.sample])[0]
                if child_prediction < parent_prediction:
                    population.append(child)
                    break

        # evaluate new individuals
        for ind in new:
            ind.fitness = objective(ind.sample)
            all_evaluations.append(ind)
            logger.log_evaluation(ind.sample, ind.fitness)
        population.extend(new)

        # select new population
        model = SurrogateModel.estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space)
        pop_predictions = model.predict([ind.sample for ind in population])
        selection_index = np.argsort(pop_predictions)[:popsize]
        population = [population[i] for i in selection_index]

    return population[0].sample, population[0].fitness

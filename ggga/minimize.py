import asyncio
import operator
import time
import typing as t

import attr
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .gpr import SurrogateModelGPR
from .util import fork_random_state, timer
from .surrogate_model import SurrogateModel
from .space import Space
from .acquisition import AcquisitionStrategy, RandomWalkAcquisition
from .individual import Individual
from .outputs import Output, OutputEventHandler


Sample = list
TimeSource = t.Callable[[], float]
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]


class OptimizationResult:
    best_individual: Individual

    def __init__(
        self, *,
        all_individuals: t.List[Individual],
        all_models: t.List[SurrogateModel],
        duration: float,
    ) -> None:
        assert all(ind.is_fully_initialized() for ind in all_individuals)

        self.all_individuals = all_individuals
        self.best_individual = min(
            all_individuals, key=lambda ind: ind.fitness)
        self.all_models = all_models
        self.duration = duration

    def best_n(self, how_many: int) -> t.List[Individual]:
        sorted_individuals = sorted(
            self.all_individuals, key=lambda ind: ind.fitness)
        return sorted_individuals[:how_many]

    @property
    def model(self) -> SurrogateModel:
        return self.all_models[-1]

    @property
    def xs(self) -> np.ndarray:
        return np.array([ind.sample for ind in self.all_individuals])

    @property
    def ys(self) -> np.ndarray:
        return np.array([ind.fitness for ind in self.all_individuals])

    @property
    def fmin(self) -> float:
        return self.best_individual.fitness


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class Minimizer:
    r"""
    Attributes
    ----------
    popsize : int
    max_nevals : int
    relscale_initial : float
    relscale_attenuation : float
    surrogate_model_class : Type[SurrogateModel]
    surrogate_model_args : dict
    acquisition_strategy : AcquisitionStrategy, optional
    time_source : TimeSource
    """

    popsize: int = 10
    max_nevals: int = 100
    relscale_initial: float = 0.3
    relscale_attenuation: float = 0.9
    surrogate_model_class: t.Type[SurrogateModel] = SurrogateModelGPR
    surrogate_model_args: dict = dict()
    acquisition_strategy: t.Optional[AcquisitionStrategy] = None
    time_source: TimeSource = time.time

    def __attrs_post_init__(self)-> None:
        assert self.popsize < self.max_nevals

    def with_setting(
        self, *,
        popsize: int = None,
        max_nevals: int = None,
        relscale_initial: float = None,
        relscale_attenuation: float = None,
        surrogate_model_class: t.Type[SurrogateModel] = None,
        surrogate_model_args: dict = None,
        acquisition_strategy: AcquisitionStrategy = None,
        time_source: TimeSource = None,
    ) -> 'Minimizer':

        TValue = t.TypeVar('TValue')

        def default(maybe: t.Optional[TValue], default: TValue) -> TValue:
            if maybe is not None:
                return maybe
            return default

        return Minimizer(
            popsize=default(popsize, self.popsize),
            max_nevals=default(max_nevals, self.max_nevals),
            relscale_initial=default(relscale_initial, self.relscale_initial),
            relscale_attenuation=default(
                relscale_attenuation, self.relscale_attenuation),
            surrogate_model_class=default(
                surrogate_model_class, self.surrogate_model_class),
            surrogate_model_args=default(
                surrogate_model_args, self.surrogate_model_args),
            acquisition_strategy=default(
                acquisition_strategy, self.acquisition_strategy),
            time_source=default(time_source, self.time_source),
        )

    async def minimize(
        self, objective: ObjectiveFunction, *,
        space: Space,
        rng: RandomState,
        outputs: OutputEventHandler = None,
    ) -> OptimizationResult:

        acquisition_strategy = self.acquisition_strategy
        if acquisition_strategy is None:
            acquisition_strategy = self._make_default_acquisition_strategy()
        assert acquisition_strategy is not None

        if outputs is None:
            outputs = Output(space=space)
        assert outputs is not None

        instance = _MinimizationInstance(
            config=self,
            objective=objective,
            space=space,
            outputs=outputs,
            acquisition_strategy=acquisition_strategy,
        )

        return await instance.run(rng=rng)

    def _make_default_acquisition_strategy(self) -> AcquisitionStrategy:
        return RandomWalkAcquisition(
            breadth=10,
            candidate_chain_length=1,
            relscale_attenuation=self.relscale_attenuation,
        )


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class _MinimizationInstance:
    config: Minimizer = attr.ib()
    objective: ObjectiveFunction = attr.ib()
    space: Space = attr.ib()
    outputs: OutputEventHandler = attr.ib()
    acquisition_strategy: AcquisitionStrategy = attr.ib()

    async def run(self, *, rng: RandomState) -> OptimizationResult:
        config: Minimizer = self.config

        total_duration = timer(config.time_source)

        population = self._make_initial_population(rng=rng)

        all_evaluations = []
        all_models = []

        for ind in population:
            ind.prediction = 0
            ind.expected_improvement = 0.0

        await self._evaluate_all(population, gen=0, rng=rng)
        all_evaluations.extend(population)

        model = self._fit_next_model(
            all_evaluations, gen=0, prev_model=None, rng=rng)
        all_models.append(model)

        generation = 0
        fmin: float = min(ind.fitness for ind in all_evaluations)
        while len(all_evaluations) < config.max_nevals:
            generation += 1
            relscale_bound = self._relscale_at_gen(generation)
            relscale = np.clip(model.length_scales(), None, relscale_bound)

            self.outputs.event_new_generation(
                generation,
                relscale=t.cast(t.Tuple[float], tuple(relscale)),
            )

            offspring = self._acquire(
                population, model=model, rng=rng, fmin=fmin, relscale=relscale)

            await self._evaluate_all(offspring, rng=rng, gen=generation)
            all_evaluations.extend(offspring)

            model = self._fit_next_model(
                all_evaluations, gen=generation, rng=rng, prev_model=model)
            all_models.append(model)

            population = self._select(parents=population, offspring=offspring)
            fmin = min(fmin, min(ind.fitness for ind in population))

        return OptimizationResult(
            all_individuals=all_evaluations,
            all_models=all_models,
            duration=total_duration(),
        )

    def _make_initial_population(
        self, *, rng: RandomState,
    ) -> t.List[Individual]:
        return [
            Individual(self.space.sample(rng=rng))
            for _ in range(self.config.popsize)
        ]

    async def _evaluate_all(
        self, individuals: t.List[Individual], *,
        rng: RandomState,
        gen: int,
    ) -> None:
        duration = timer(self.config.time_source)

        results = await asyncio.gather(*(
            self.objective(ind.sample, fork_random_state(rng))
            for ind in individuals
        ))

        for ind, (fitness, cost) in zip(individuals, results):
            ind.fitness = fitness
            ind.cost = cost
            ind.gen = gen
            assert ind.is_fully_initialized(), repr(ind)

        self.outputs.event_evaluations_completed(
            individuals, duration=duration())

    def _fit_next_model(
        self, all_evaluations: t.List[Individual], *,
        gen: int,
        rng: RandomState,
        prev_model: t.Optional[SurrogateModel],
    ) -> SurrogateModel:
        duration = timer(self.config.time_source)

        model = t.cast(t.Any, self.config.surrogate_model_class).estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=self.space, rng=rng, prior=prev_model,
            **self.config.surrogate_model_args,
        )

        self.outputs.event_model_trained(gen, model, duration=duration())

        return model

    def _relscale_at_gen(self, gen: int) -> float:
        assert gen >= 1, f"gen must be positive: {gen}"
        attenuation = self.config.relscale_attenuation ** (gen - 1)
        return attenuation * self.config.relscale_initial

    def _acquire(
        self, population: t.List[Individual], *,
        model: SurrogateModel,
        rng: RandomState,
        fmin: float,
        relscale: np.ndarray,
    ) -> t.List[Individual]:
        duration = timer(self.config.time_source)

        offspring = list(self.acquisition_strategy.acquire(
            population, model=model, space=self.space,
            rng=rng, fmin=fmin, relscale=relscale))

        self.outputs.event_acquisition_completed(duration=duration())

        return offspring

    def _select(  # pylint: disable=no-self-use
        self, *,
        parents: t.List[Individual],
        offspring: t.List[Individual],
    ) -> t.List[Individual]:
        selected, rejected = select_next_population(
            parents=parents, offspring=offspring)

        population = replace_worst_n_individuals(
            3, population=selected, replacement_pool=rejected)

        return population


def select_next_population(
    *, parents: t.List[Individual], offspring: t.List[Individual],
) -> t.Tuple[t.List[Individual], t.List[Individual]]:
    r"""
    Select the offspring, unless the corresponding parent was better.
    This avoids moving the search into “worse” areas,
    although this doesn't consider the acquisition strategy (EI).
    So this is a greedy hill-climbing approach based purely on the observed
    (possibly fuzzy) fitness value.
    """
    assert len(parents) == len(offspring)

    selected = []
    rejected = []

    for ind_parent, ind_offspring in zip(parents, offspring):
        select = ind_offspring
        reject = ind_parent

        if reject.fitness < select.fitness:
            select, reject = reject, select

        selected.append(select)
        rejected.append(reject)

    return selected, rejected


def replace_worst_n_individuals(
    replace_worst_n: int, *,
    population: t.List[Individual],
    replacement_pool: t.List[Individual],
) -> t.List[Individual]:
    r"""
    Allow the top N individuals from the replacement pool
    to become part of the population, if they have better fitness.
    """
    by_fitness = operator.attrgetter('fitness')
    candidates = sorted(replacement_pool, key=by_fitness)[:replace_worst_n]
    chosen = sorted(population + candidates, key=by_fitness)[:len(population)]
    return chosen


__all__ = [
    Minimizer.__name__,
    OptimizationResult.__name__,
]

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
from .acquisition import AcquisitionStrategy, MutationAcquisition
from .individual import Individual
from .outputs import Output, OutputEventHandler


Sample = list
TimeSource = t.Callable[[], float]
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]


class OptimizationResult:
    """Results of one optimization run (multiple experiments)."""

    best_individual: Individual

    def __init__(
        self, *,
        all_individuals: t.List[Individual],
        all_models: t.List[SurrogateModel],
        duration: float,
    ) -> None:
        assert all(ind.is_fully_initialized() for ind in all_individuals)

        #: list[Individual]: all results
        self.all_individuals = all_individuals
        #: Individual: best result
        self.best_individual = min(
            all_individuals, key=lambda ind: ind.observation)
        #: list[SurrogateModel]: all models
        self.all_models = all_models
        #: float: total duration
        self.duration = duration

    def best_n(self, how_many: int) -> t.List[Individual]:
        """Select the best evaluation results."""
        sorted_individuals = sorted(
            self.all_individuals, key=lambda ind: ind.observation)
        return sorted_individuals[:how_many]

    @property
    def model(self) -> SurrogateModel:
        """Final model."""
        return self.all_models[-1]

    @property
    def xs(self) -> np.ndarray:
        """Input variables of all evaluations."""
        return np.array([ind.sample for ind in self.all_individuals])

    @property
    def ys(self) -> np.ndarray:
        """Output variables of all evaluations."""
        return np.array([ind.observation for ind in self.all_individuals])

    @property
    def fmin(self) -> float:
        """Best observed value."""
        return self.best_individual.observation


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class Minimizer:
    """Configure the GGGA optimizer.

    Parameters
    ----------
    popsize: int, optional
        How many samples are taken per generation.
        Defaults to 10.
    initial: int, optional
        How many initial samples should be acquired
        before model-guided acquisition takes over.
        Default: 10.
    max_nevals: int, optional
        How many samples may be taken in total per optimization run.
        Defaults to 100.
    relscale_initial: float, optional
        Standard deviation for creating new samples,
        as percentage of each paramter's range.
        Defaults to 0.3.
    relscale_attenuation: float, optional
        Factor by which the relscale is reduced per generation.
        Defaults to 0.9.
    surrogate_model_class: type[SurrogateModel], optional
        The regression model to fit the response surface.
        Defaults to :class:`~ggga.gpr.SurrogateModelGPR`.
    surrogate_model_args: dict, optional
        Extra arguments for the surrogate model.
    acquisition_strategy: AcquisitionStrategy or None, optional
        How new samples are acquired.
        Defaults to :class:`~ggga.acquisition.MutationAcquisition`
        with breadth=10.
    select_via_posterior: bool, optional
        Whether the model prediction should be used as a fitness function
        when selecting which samples proceed to the next generation.
        If false, the objective's observed value incl. noise is used.
        Defaults to False.
    fmin_via_posterior: bool, optional
        Whether the model prediction is used
        to find the current best point during optimization.
        If false, the objective's observed value incl. noise is used.
        Defaults to True.
    n_replacements: int, optional
        How many random samples are suggested per generation.
        Usually, new samples are created by random mutations
        of existing samples.
    """

    popsize: int = 10
    initial: int = 10
    max_nevals: int = 100
    relscale_initial: float = 0.3
    relscale_attenuation: float = 0.9
    surrogate_model_class: t.Type[SurrogateModel] = SurrogateModelGPR
    surrogate_model_args: dict = dict()
    acquisition_strategy: t.Optional[AcquisitionStrategy] = None
    time_source: TimeSource = time.time
    select_via_posterior: bool = False
    fmin_via_posterior: bool = True
    n_replacements: int = 1

    def __attrs_post_init__(self)-> None:
        assert self.initial + self.popsize <= self.max_nevals, \
            f"evaluation budget {self.max_nevals} to small" \
            f"with {self.initial}+n*{self.popsize} evaluations"

    def with_setting(
        self, *,
        popsize: int = None,
        initial: int = None,
        max_nevals: int = None,
        relscale_initial: float = None,
        relscale_attenuation: float = None,
        surrogate_model_class: t.Type[SurrogateModel] = None,
        surrogate_model_args: dict = None,
        acquisition_strategy: AcquisitionStrategy = None,
        time_source: TimeSource = None,
        select_via_posterior: bool = None,
        fmin_via_posterior: bool = None,
        n_replacements: int = None,
    ) -> 'Minimizer':
        """Clone a Minimizer but override some attributes."""

        TValue = t.TypeVar('TValue')

        def default(maybe: t.Optional[TValue], default: TValue) -> TValue:
            if maybe is not None:
                return maybe
            return default

        return Minimizer(
            popsize=default(popsize, self.popsize),
            initial=default(initial, self.initial),
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
            select_via_posterior=default(
                select_via_posterior, self.select_via_posterior),
            fmin_via_posterior=default(
                fmin_via_posterior, self.fmin_via_posterior),
            n_replacements=default(n_replacements, self.n_replacements)
        )

    async def minimize(
        self, objective: ObjectiveFunction, *,
        space: Space,
        rng: RandomState,
        outputs: OutputEventHandler = None,
        historic_individuals: t.Iterable[Individual] = (),
    ) -> OptimizationResult:
        """Minimize the objective.

        Parameters
        ----------
        objective: ``async objective(sample, rng) -> (value, cost)``)
            A function to calculate the objective value.
            The *sample* is a list
            with the same order as the params in the space.
            The *value* and *cost* are floats.
            The cost is merely informative.
        space
            The parameter space inside which the objective is optimized.
        rng
        outputs
            Controls what information is printed during optimization.
            Can e.g. be used to save evaluations into a CSV file.
            Defaults to :class:`~ggga.outputs.Output`.
        historic_individuals
            Previous evaluations of the same objective/space
            that should be incorporated into the model.
            Can be useful in order to benefit from previous minimization runs.
            Potential drawbacks include a biased model,
            and that the tuner slows down with additional samples.
            Constraints: all individual must be fully initialized,
            and declare the -1th generation.
        """

        acquisition_strategy = self.acquisition_strategy
        if acquisition_strategy is None:
            acquisition_strategy = MutationAcquisition(breadth=10)
        assert acquisition_strategy is not None

        if outputs is None:
            outputs = Output(space=space)
        assert outputs is not None

        assert all(ind.is_fully_initialized() for ind in historic_individuals)
        assert all(ind.gen == -1 for ind in historic_individuals)

        instance = _MinimizationInstance(
            config=self,
            objective=objective,
            space=space,
            outputs=outputs,
            acquisition_strategy=acquisition_strategy,
        )

        return await instance.run(
            rng=rng,
            historic_individuals=historic_individuals)


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class _MinimizationInstance:
    config: Minimizer = attr.ib()
    objective: ObjectiveFunction = attr.ib()
    space: Space = attr.ib()
    outputs: OutputEventHandler = attr.ib()
    acquisition_strategy: AcquisitionStrategy = attr.ib()

    async def run(
            self, *,
            rng: RandomState,
            historic_individuals: t.Iterable[Individual],
    ) -> OptimizationResult:
        config: Minimizer = self.config

        total_duration = timer(config.time_source)

        population = self._make_initial_population(rng=rng)

        budget = config.max_nevals
        all_evaluations: t.List[Individual] = []
        all_models = []

        all_evaluations.extend(historic_individuals)

        for ind in population:
            ind.prediction = 0
            ind.expected_improvement = 0.0

        await self._evaluate_all(population, gen=0, rng=rng)
        budget -= len(population)
        all_evaluations.extend(population)

        model = self._fit_next_model(
            all_evaluations, gen=0, prev_model=None, rng=rng)
        all_models.append(model)

        def find_fmin(
            individuals: t.Iterable[Individual], *, model: SurrogateModel,
        ) -> float:
            fmin_operator = self._make_fitness_operator(
                with_posterior=self.config.fmin_via_posterior,
                model=model)
            return min(fmin_operator(ind) for ind in individuals)

        fmin: float = find_fmin(all_evaluations, model=model)

        generation = 0
        while budget > 0:
            generation += 1
            population = self._resize_population(
                population, min(budget, config.popsize), model=model, rng=rng)
            relscale_bound = self._relscale_at_gen(generation)
            relscale = np.clip(model.length_scales(), None, relscale_bound)

            self.outputs.event_new_generation(
                generation,
                relscale=t.cast(t.Tuple[float], tuple(relscale)),
            )

            offspring = self._acquire(
                population, model=model, rng=rng, fmin=fmin, relscale=relscale)

            await self._evaluate_all(offspring, rng=rng, gen=generation)
            budget -= len(offspring)
            all_evaluations.extend(offspring)

            model = self._fit_next_model(
                all_evaluations, gen=generation, rng=rng, prev_model=model)
            all_models.append(model)

            population = self._select(
                parents=population, offspring=offspring, model=model)
            fmin = find_fmin(all_evaluations, model=model)

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
            for _ in range(self.config.initial)
        ]

    @staticmethod
    def _make_fitness_operator(
        *, with_posterior: bool, model: SurrogateModel,
    ) -> t.Callable[[Individual], float]:

        if with_posterior:
            def fitness(ind: Individual) -> float:
                y, _std = model.predict(ind.sample)
                return y
        else:
            def fitness(ind: Individual) -> float:
                return ind.observation

        return fitness

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

        for ind, (observation, cost) in zip(individuals, results):
            ind.observation = observation
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
            [ind.observation for ind in all_evaluations],
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
        model: SurrogateModel,
    ) -> t.List[Individual]:

        fitness_operator = self._make_fitness_operator(
            with_posterior=self.config.select_via_posterior, model=model)

        selected, rejected = select_next_population(
            parents=parents, offspring=offspring, fitness=fitness_operator)

        population = replace_worst_n_individuals(
            self.config.n_replacements,
            population=selected,
            replacement_pool=rejected)

        return population

    def _resize_population(
            self, population: t.List[Individual], newsize: int, *,
            model: SurrogateModel,
            rng: RandomState,
    ) -> t.List[Individual]:
        # Sort the individuals.
        fitness_operator = self._make_fitness_operator(
            with_posterior=self.config.select_via_posterior, model=model)
        population = sorted(population, key=fitness_operator)

        # If there are too many individuals, remove worst-ranking individuals.
        if len(population) > newsize:
            return population[:newsize]

        # If there are too few individuals, repeat individuals.
        # It might be better to pad with random choices, but these individuals
        # must be fully evaluated.
        pool = list(population)
        while len(population) < newsize:
            population.append(pool[rng.randint(len(pool))])
        return population


def select_next_population(
    *,
    parents: t.List[Individual],
    offspring: t.List[Individual],
    fitness: t.Callable[[Individual], float],
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

        if fitness(reject) < fitness(select):
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
    by_fitness = operator.attrgetter('observation')
    candidates = sorted(replacement_pool, key=by_fitness)[:replace_worst_n]
    chosen = sorted(population + candidates, key=by_fitness)[:len(population)]
    return chosen


__all__ = [
    Minimizer.__name__,
    OptimizationResult.__name__,
]

import asyncio
import csv
import json
import operator
import sys
import time
import typing as t

import attr
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

from .gpr import SurrogateModelGPR
from .util import tabularize, fork_random_state, timer
from .surrogate_model import SurrogateModel
from .space import Space, Real
from .acquisition import (
        AcquisitionStrategy,
        ChainedAcquisition,
        RandomWalkAcquisition,
        RandomReplacementAcquisition)
from .individual import Individual


Sample = list
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]
CsvWriterow = t.Callable[[t.Iterable], None]


class IndividualsToTable:
    def __init__(self, space: Space) -> None:
        self.columns: t.List[str] = []
        self.columns.extend(
            'gen utility prediction ei cost'.split())
        self.columns.extend(
            f"param_{param.name}" for param in space.params)

        self.formats: t.List[str] = []
        self.formats.extend(
            '{:2d} {:.2f} {:.2f} {:.2e} {:.2f}'.split())
        self.formats.extend(
            '{:.5f}' if isinstance(p, Real) else '{}'
            for p in space.params)

    @staticmethod
    def individual_to_row(ind: Individual) -> t.Iterable:
        yield from [ind.gen, ind.fitness, ind.prediction, ind.ei, ind.cost]
        yield from ind.sample


class LoggerInterface():
    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        pass

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        pass

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        pass

    def event_acquisition_completed(
        self, *, duration: float,
    ) -> None:
        pass


class Subloggers(LoggerInterface):
    def __init__(self, *subloggers: LoggerInterface) -> None:
        self.subloggers = list(subloggers)

    def add(self, logger: LoggerInterface) -> None:
        self.subloggers.append(logger)

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        for logger in self.subloggers:
            logger.event_new_generation(gen, relscale=relscale)

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_evaluations_completed(individuals, duration=duration)

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_model_trained(generation, model, duration=duration)

    def event_acquisition_completed(
        self, *, duration: float,
    ) -> None:
        for logger in self.subloggers:
            logger.event_acquisition_completed(duration=duration)


class ModelRecordingLogger(LoggerInterface):
    def __init__(self, model_file: t.TextIO) -> None:
        self._model_file = model_file

        assert hasattr(model_file, 'write'), \
            f"Model output must be a writable file object: {model_file!r}"

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *,
        duration: float,  # pylint: disable=unused-argument
    ) -> None:

        json.dump(
            [generation, model.to_jsonish()],
            self._model_file,
            default=self._coerce_to_jsonish)
        print(file=self._model_file)

    @staticmethod
    def _coerce_to_jsonish(some_object):
        if isinstance(some_object, np.ndarray):
            return list(some_object)
        raise TypeError(f"cannot encode as JSON: {some_object!r}")


class HumanReadableLogger(LoggerInterface):
    def __init__(
        self, *,
        log_file: t.TextIO,
        individuals_table: IndividualsToTable,
    ) -> None:
        self._file = log_file
        self._individuals_table = individuals_table

    def log(self, msg: str, *, level: str = 'INFO') -> None:
        assert level == 'INFO'
        marker = f"[{level}]"

        first = True
        for line in msg.splitlines():
            if first:
                print(marker, line, file=self._file)
                first = False
            else:
                print(" " * len(marker), line, file=self._file)

    def event_new_generation(
        self, gen: int, *, relscale: t.Tuple[float],
    ) -> None:
        formatted_relscale = ' '.join(format(r, '.5') for r in relscale)
        self.log(f"starting generation #{gen} (relscale {formatted_relscale})")

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *, duration: float,
    ) -> None:
        self.log(f'evaluations ({duration} s):\n' + tabularize(
            header=self._individuals_table.columns,
            formats=self._individuals_table.formats,
            data=[
                list(self._individuals_table.individual_to_row(ind))
                for ind in sorted(individuals,
                                  key=operator.attrgetter('fitness'))],
        ))

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *, duration: float,
    ) -> None:
        self.log(
            f"trained new model ({duration} s):\n"
            f"{model!r}")


class EvaluationsWritingLogger(LoggerInterface):
    def __init__(
        self, csv_file: t.TextIO, *, individuals_table: IndividualsToTable,
    ) -> None:
        assert hasattr(csv_file, 'write'), \
            f"Evaluation CSV file must be a writable file object: {csv_file!r}"
        self._csv_writer = csv.writer(csv_file)
        self._csv_writer.writerow(individuals_table.columns)
        self._individuals_table = individuals_table

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *,
        duration: float,  # pylint: disable=unused-argument
    ) -> None:
        for ind in individuals:
            self._csv_writer.writerow(
                self._individuals_table.individual_to_row(ind))


class Logger(LoggerInterface):
    r"""
    Control the output during optimization.

    Attributes:
        space (Space):
            The parameter space.
        evaluation_csv_file (TextIO, optional):
            If present, all evaluations are recorded in this file.
        model_file (TextIO, optional):
            If present, metadata of the models is recorded in this file,
            using a JSON-per-line format.
        log_file (TextIO, optional):
            Where to write human-readable logs. Defaults to sys.stdout.
            If set to None, output is suppressed.
    """

    def __init__(
        self, *,
        space: Space,
        evaluation_csv_file: t.Optional[t.TextIO] = None,
        model_file: t.Optional[t.TextIO] = None,
        log_file: t.Optional[t.TextIO] = sys.stdout,
    ) -> None:

        individuals_table = IndividualsToTable(space)

        self.space: Space = space
        self._subloggers = Subloggers()

        if log_file is not None:
            self._subloggers.add(HumanReadableLogger(
                log_file=log_file, individuals_table=individuals_table))

        if model_file is not None:
            self._subloggers.add(ModelRecordingLogger(model_file))

        if evaluation_csv_file is not None:
            self._subloggers.add(EvaluationsWritingLogger(
                evaluation_csv_file, individuals_table=individuals_table))

        # durations
        self.acquisition_durations: t.List[float] = []
        self.evaluation_durations: t.List[float] = []
        self.training_durations: t.List[float] = []

    def event_evaluations_completed(
        self, individuals: t.Iterable[Individual], *,
        duration: float,
    ) -> None:
        self.evaluation_durations.append(duration)

        self._subloggers.event_evaluations_completed(
            individuals, duration=duration)

    def event_model_trained(
        self, generation: int, model: SurrogateModel, *,
        duration: float,
    ) -> None:
        self.training_durations.append(duration)

        self._subloggers.event_model_trained(
            generation, model, duration=duration)

    def event_new_generation(
        self, gen: int, *,
        relscale: t.Tuple[float],
    ) -> None:
        self._subloggers.event_new_generation(gen, relscale=relscale)

    def event_acquisition_completed(self, *, duration: float) -> None:
        self.acquisition_durations.append(duration)

        self._subloggers.event_acquisition_completed(duration=duration)


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


TimeSource = t.Callable[[], float]


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class Minimizer:
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
        logger: t.Optional[LoggerInterface],
    ) -> OptimizationResult:

        acquisition_strategy = self.acquisition_strategy
        if acquisition_strategy is None:
            acquisition_strategy = self._make_default_acquisition_strategy(
                space)
        assert acquisition_strategy is not None

        if logger is None:
            logger = Logger(space=space)
        assert logger is not None

        instance = _MinimizationInstance(
            config=self,
            objective=objective,
            space=space,
            logger=logger,
            acquisition_strategy=acquisition_strategy,
        )

        return await instance.run(rng=rng)

    def _make_default_acquisition_strategy(
        self, space: Space,
    ) -> AcquisitionStrategy:
        return ChainedAcquisition(
            RandomWalkAcquisition(
                breadth=3,
                candidate_chain_length=1,
                relscale_attenuation=self.relscale_attenuation,
                space=space,
            ),
            RandomReplacementAcquisition(
                n_replacements=self.popsize, space=space,
            ),
        )


@attr.s(frozen=True, cmp=False, auto_attribs=True)
class _MinimizationInstance:
    config: Minimizer = attr.ib()
    objective: ObjectiveFunction = attr.ib()
    space: Space = attr.ib()
    logger: LoggerInterface = attr.ib()
    acquisition_strategy: AcquisitionStrategy = attr.ib()

    async def run(self, *, rng: RandomState) -> OptimizationResult:
        config: Minimizer = self.config

        total_duration = timer(config.time_source)

        population = self._make_initial_population(rng=rng)

        all_evaluations = []
        all_models = []

        for ind in population:
            ind.prediction = 0
            ind.ei = 0.0

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

            self.logger.event_new_generation(
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

        self.logger.event_evaluations_completed(
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

        self.logger.event_model_trained(gen, model, duration=duration())

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
            population, model=model, rng=rng, fmin=fmin, relscale=relscale))

        self.logger.event_acquisition_completed(duration=duration())

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


async def minimize(  # pylint: disable=too-many-locals
    objective: ObjectiveFunction,
    *,
    space: Space,
    popsize: int = 10,
    max_nevals: int = 100,
    logger: LoggerInterface = None,
    rng: RandomState,
    relscale_initial: float = 0.3,
    relscale_attenuation: float = 0.9,
    surrogate_model_class: t.Type[SurrogateModel] = SurrogateModelGPR,
    surrogate_model_args: dict = dict(),
    acquisition_strategy: AcquisitionStrategy = None,
    time_source: t.Callable[[], float] = time.time
) -> OptimizationResult:

    minimizer = Minimizer().with_setting(
        popsize=popsize,
        max_nevals=max_nevals,
        relscale_initial=relscale_initial,
        relscale_attenuation=relscale_attenuation,
        surrogate_model_class=surrogate_model_class,
        surrogate_model_args=surrogate_model_args,
        acquisition_strategy=acquisition_strategy,
        time_source=time_source,
    )

    return await minimizer.minimize(
        objective, space=space, rng=rng, logger=logger)


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

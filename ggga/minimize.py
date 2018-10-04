import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import typing as t
import asyncio
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
import time
import attr
import csv
import operator
import json


Sample = list
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]


@attr.s
class Logger:
    space: Space = attr.ib()
    evaluation_csv_file: t.Optional[t.TextIO] = attr.ib()
    model_file: t.Optional[t.TextIO] = attr.ib()

    acquisition_durations: t.List[float] = attr.ib(init=False, factory=list)
    evaluation_durations: t.List[float] = attr.ib(init=False, factory=list)
    training_durations: t.List[float] = attr.ib(init=False, factory=list)

    _evaluation_csv_writerow: t.Optional[t.Callable[[t.Iterable], None]] = \
        attr.ib(init=False, default=None)
    _evaluation_csv_columns: t.List[str] = attr.ib(init=False)
    _evaluation_log_formats: t.List[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._evaluation_csv_columns = []
        self._evaluation_csv_columns.extend(
            'gen utility prediction ei cost'.split())
        self._evaluation_csv_columns.extend(
            f"param_{param.name}" for param in self.space.params)

        self._evaluation_log_formats = []
        self._evaluation_log_formats.extend(
            '{:2d} {:.2f} {:.2f} {:.2e} {:.2f}'.split())
        self._evaluation_log_formats.extend(
            '{:.5f}' if isinstance(p, Real) else '{}'
            for p in self.space.params)

        if self.evaluation_csv_file is not None:
            assert hasattr(self.evaluation_csv_file, 'write'), (
                f"Evaluation CSV output must be writable file object: "
                f"{self.evaluation_csv_file!r}")
            self._evaluation_csv_writerow = \
                csv.writer(self.evaluation_csv_file).writerow
            self._evaluation_csv_writerow(self._evaluation_csv_columns)

        if self.model_file is not None:
            assert hasattr(self.model_file, 'write'), (
                f"Model output must be writable file object: ",
                f"{self.model_file!r}")

    def _get_csv_row(self, ind: Individual) -> t.Iterable:
        yield from [ind.gen, ind.fitness, ind.prediction, ind.ei, ind.cost]
        yield from ind.sample

    def log(self, msg: str, *, level: str = 'INFO') -> None:
        marker = f"[{level}]"
        first = True
        for line in msg.splitlines():
            if first:
                print(marker, line)
                first = False
            else:
                print(" " * len(marker), line)

    def record_evaluations(
        self, individuals: t.Iterable[Individual], *,
        duration: float,
    ) -> None:
        self.evaluation_durations.append(duration)

        self.log(f'evaluations ({duration} s):\n' + tabularize(
            header=self._evaluation_csv_columns,
            formats=self._evaluation_log_formats,
            data=[
                list(self._get_csv_row(ind))
                for ind in sorted(individuals,
                                  key=operator.attrgetter('fitness'))],
        ))

        if self._evaluation_csv_writerow is not None:
            for ind in individuals:
                self._evaluation_csv_writerow(self._get_csv_row(ind))

    def record_model(
        self, generation: int, model: SurrogateModel, *,
        duration: float,
    ) -> None:
        self.training_durations.append(duration)

        self.log(
            f"trained new model ({duration} s):\n"
            f"{model!r}")

        def default(o):
            if isinstance(o, np.ndarray):
                return list(o)
            raise TypeError(f"cannot encode as JSON: {o!r}")

        if self.model_file is not None:
            json.dump(
                [generation, model.to_jsonish()], self.model_file,
                default=default)
            print(file=self.model_file)

    def announce_new_generation(
        self, gen: int, *,
        relscale: t.Tuple[float],
    ) -> None:
        formatted_relscale = ' '.join(format(r, '.5') for r in relscale)
        self.log(f"starting generation #{gen} (relscale {formatted_relscale})")

    def record_acquisition(self, *, duration: float) -> None:
        self.acquisition_durations.append(duration)


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

    def best_n(self, n: int) -> t.List[Individual]:
        return sorted(self.all_individuals, key=lambda ind: ind.fitness)[:n]

    @property
    def fmin(self) -> float:
        return self.best_individual.fitness


async def minimize(
    objective: ObjectiveFunction,
    *,
    space: Space,
    popsize: int=10,
    max_nevals: int=100,
    logger: Logger=None,
    rng: RandomState,
    relscale_initial: float = 0.3,
    relscale_attenuation: float = 0.9,
    surrogate_model_class: t.Type[SurrogateModel] = SurrogateModelGPR,
    surrogate_model_args: dict=dict(),
    acquisition_strategy: AcquisitionStrategy = None,
    evaluation_csv_file: t.TextIO = None,
    model_file: t.TextIO = None,
    time_source: t.Callable[[], float] = time.time
) -> OptimizationResult:

    if logger is None:
        logger = Logger(
            space=space,
            evaluation_csv_file=evaluation_csv_file,
            model_file=model_file,
        )

    if acquisition_strategy is None:
        acquisition_strategy = ChainedAcquisition(
            RandomWalkAcquisition(
                breadth=3,
                candidate_chain_length=1,
                relscale_attenuation=relscale_attenuation,
                space=space,
            ),
            RandomReplacementAcquisition(
                n_replacements=popsize, space=space,
            ),
        )

    assert popsize < max_nevals

    total_duration = timer(time_source)

    async def evaluate_all(
        individuals: t.List[Individual], *,
        rng: RandomState,
        gen: int,
    ) -> float:
        duration = timer(time_source)

        results = await asyncio.gather(*(
            objective(ind.sample, fork_random_state(rng))
            for ind in individuals))

        for ind, (fitness, cost) in zip(individuals, results):
            ind.fitness = fitness
            ind.cost = cost
            ind.gen = gen
            assert ind.is_fully_initialized(), repr(ind)

        return duration()

    def fit_next_model(
        all_evaluations, *, rng, prev_model,
    ) -> t.Tuple[float, SurrogateModel]:
        duration = timer(time_source)
        model = t.cast(t.Any, surrogate_model_class).estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=space, rng=rng, prior=prev_model, **surrogate_model_args)
        return duration(), model

    population = [Individual(space.sample(rng=rng)) for _ in range(popsize)]

    all_evaluations = []
    all_models = []

    for ind in population:
        ind.prediction = 0
        ind.ei = 0.0

    evaluation_duration = await evaluate_all(population, rng=rng, gen=0)
    logger.record_evaluations(population, duration=evaluation_duration)
    all_evaluations.extend(population)

    fitting_duration, model = fit_next_model(
        all_evaluations, rng=rng, prev_model=None)
    all_models.append(model)
    logger.record_model(0, model, duration=fitting_duration)

    generation = 0
    fmin: float = min(ind.fitness for ind in all_evaluations)
    while len(all_evaluations) < max_nevals:
        generation += 1
        relscale_bound = \
            relscale_initial * (relscale_attenuation**(generation - 1))
        relscale = np.clip(model.length_scales(), None, relscale_bound)

        logger.announce_new_generation(
            generation,
            relscale=t.cast(t.Tuple[float], tuple(relscale)),
        )

        acquisition_duration = timer(time_source)

        offspring = list(acquisition_strategy.acquire(
            population,
            model=model,
            rng=rng,
            fmin=fmin,
            relscale=relscale,
        ))

        logger.record_acquisition(duration=acquisition_duration())

        # evaluate new individuals
        evaluation_duration = await evaluate_all(
            offspring, rng=rng, gen=generation)
        all_evaluations.extend(offspring)
        logger.record_evaluations(offspring, duration=evaluation_duration)

        fitting_duration, model = fit_next_model(
            all_evaluations, rng=rng, prev_model=model)
        all_models.append(model)
        logger.record_model(generation, model, duration=fitting_duration)

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
        fmin = min(fmin, min(ind.fitness for ind in population))

    return OptimizationResult(
        all_individuals=all_evaluations,
        all_models=all_models,
        duration=total_duration(),
    )

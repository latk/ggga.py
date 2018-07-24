import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import typing as t
import asyncio
from .gpr import SurrogateModelGPR
from .util import tabularize, fork_random_state
from .surrogate_model import SurrogateModel
from .space import Space, Real
from .acquisition import (
        AcquisitionStrategy,
        ChainedAcquisition,
        RandomWalkAcquisition,
        RandomReplacementAcquisition)
from .individual import Individual
import time


Sample = list
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]


class Logger(object):
    def record_evaluations(self, individuals, *, space):
        print("[INFO] evaluations:")
        param_names = [p.name for p in space.params]
        param_formats = \
            ['{:.5f}' if isinstance(p, Real) else '{}' for p in space.params]
        print(tabularize(
            header=['utility', 'prediction', 'ei', 'cost', *param_names],
            formats=['{:.2f}', '{:.2f}', '{:.2e}', '{:.2f}', *param_formats],
            data=[[ind.fitness, ind.prediction, ind.ei, ind.cost, *ind.sample]
                  for ind in individuals],
        ))

    def announce_new_generation(
        self, gen: int, *,
        model: SurrogateModel,
        relscale: t.Tuple[float],
    ) -> None:
        formatted_relscale = ' '.join(format(r, '.5') for r in relscale)
        print(f"[INFO] starting generation #{gen}")
        print(f"       relscale ({formatted_relscale})")
        print(f"       estimator: {model!r}")


class OptimizationResult(object):
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


async def minimize(
    objective: ObjectiveFunction,
    *,
    space: Space,
    popsize: int=10,
    max_nevals: int=100,
    logger: Logger=None,
    rng: RandomState,
    relscale_initial=0.3,
    relscale_attenuation=0.9,
    surrogate_model_class: t.Type[SurrogateModel] = SurrogateModelGPR,
    surrogate_model_args: dict=dict(),
    acquisition_strategy: AcquisitionStrategy = None,
) -> OptimizationResult:

    if logger is None:
        logger = Logger()

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

    start_time = time.time()

    async def evaluate_all(
        individuals: t.List[Individual], *,
        rng: RandomState,
        gen: int,
    ) -> None:
        results = await asyncio.gather(*(
            objective(ind.sample, fork_random_state(rng))
            for ind in individuals))
        for ind, (fitness, cost) in zip(individuals, results):
            ind.fitness = fitness
            ind.cost = cost
            ind.gen = gen
            assert ind.is_fully_initialized(), repr(ind)

    def fit_next_model(all_evaluations, rng, prev_model):
        return surrogate_model_class.estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=space, rng=rng, prior=prev_model, **surrogate_model_args)

    population = [Individual(space.sample(rng=rng)) for _ in range(popsize)]

    all_evaluations = []
    all_models = []

    for ind in population:
        ind.prediction = 0
        ind.ei = 0.0

    await evaluate_all(population, rng=rng, gen=0)
    logger.record_evaluations(population, space=space)
    all_evaluations.extend(population)

    model = fit_next_model(all_evaluations, rng=rng, prev_model=None)
    all_models.append(model)

    generation = 0
    fmin: float = min(ind.fitness for ind in all_evaluations)
    while len(all_evaluations) < max_nevals:
        generation += 1
        relscale_bound = \
            relscale_initial * (relscale_attenuation**(generation - 1))
        relscale = np.clip(model.length_scales(), None, relscale_bound)

        logger.announce_new_generation(
            generation,
            model=model,
            relscale=t.cast(t.Tuple[float], tuple(relscale)),
        )

        offspring = list(acquisition_strategy.acquire(
            population,
            model=model,
            rng=rng,
            fmin=fmin,
            relscale=relscale,
        ))

        # evaluate new individuals
        await evaluate_all(offspring, rng=rng, gen=generation)
        all_evaluations.extend(offspring)
        logger.record_evaluations(offspring, space=space)

        model = fit_next_model(all_evaluations, rng=rng, prev_model=model)
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
        fmin = min(fmin, min(ind.fitness for ind in population))

    total_duration = time.time() - start_time

    return OptimizationResult(
        all_individuals=all_evaluations,
        all_models=all_models,
        duration=total_duration,
    )

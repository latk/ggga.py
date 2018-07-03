import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import scipy.stats  # type: ignore
import typing as t
import asyncio
from .gpr import SurrogateModelGPR
from .util import tabularize, fork_random_state
from .surrogate_model import SurrogateModel
from .space import Space, Real
import time


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
        print("       estimator: {!r}".format(model))


class Individual(object):
    sample: list
    fitness: float

    def __init__(self, sample, fitness):
        self.sample = sample
        self.fitness = fitness


def expected_improvement(mean, std, fmin):
    norm = scipy.stats.norm
    z = -(mean - fmin) / std
    ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z)
    return ei


class OptimizationResult(object):
    best_individual: Individual

    def __init__(
        self, *,
        all_individuals: t.List[Individual],
        all_models: t.List[SurrogateModel],
        duration: float,
    ) -> None:
        self.all_individuals = all_individuals
        self.best_individual = min(
            all_individuals, key=lambda ind: ind.fitness)
        self.all_models = all_models
        self.duration = duration

    def best_n(self, n: int) -> t.List[Individual]:
        return sorted(self.all_individuals, key=lambda ind: ind.fitness)[:n]


def suggest_next_candidate(
    parent_sample, *, relscale, fmin, rng, breadth, model, space,
) -> t.Tuple[list, float]:
    candidate_samples = [
        space.mutate(parent_sample, relscale=relscale, rng=rng)
        for _ in range(breadth)]
    candidate_mean, candidate_std = model.predict_a(candidate_samples)
    candidate_ei = expected_improvement(
        candidate_mean, candidate_std, fmin)
    candidate_index = np.argmax(candidate_ei)
    return (candidate_samples[candidate_index],
            candidate_ei[candidate_index])


def suggest_next_candidate_a(
    parent_samples, *, relscale, fmin, rng, breadth, model, space,
) -> t.Tuple[t.List[list], t.List[float]]:
    samples: t.List[list] = t.cast(list, [None] * len(parent_samples))
    eis: t.List[float] = t.cast(list, [None] * len(parent_samples))
    for i in range(len(parent_samples)):
        samples[i], eis[i] = suggest_next_candidate(
            parent_samples[i],
            relscale=relscale, fmin=fmin, rng=rng,
            breadth=breadth, model=model, space=space)
    return samples, eis


def improve_through_random_replacement(
    offspring_samples: t.List[list],
    *,
    offspring_ei: t.List[float],
    offspring_ey: t.Optional[t.List[float]],
    model: SurrogateModel,
    space: Space,
    fmin: float,
    rng: RandomState,
    n_replacements: int,
    n_suggestion_iters: int = 3,
    n_suggestion_breadth: int = 5,
    relscale_initial: float = 0.1,
    relscale_attenuation: float = 0.5,
) -> t.Iterable[list]:

    # rr = random replacement
    rr_samples = [space.sample(rng=rng) for _ in range(n_replacements)]
    rr_ei = [0.0] * n_replacements

    assert n_suggestion_iters > 0
    for i in range(n_suggestion_iters):
        relscale = relscale_initial * relscale_attenuation**i
        rr_samples, rr_ei = suggest_next_candidate_a(
            rr_samples,
            relscale=relscale, fmin=fmin,
            rng=rng, model=model, space=space,
            breadth=n_suggestion_breadth)

    if offspring_ey is not None:
        rr_ey = model.predict_a(rr_samples, return_std=False)

    # indices will be popped from the queue worst to best
    rr_index_queue = list(np.argsort(rr_ei)[::-1])

    # search offspring from worst to best
    for offspring_i in np.argsort(offspring_ei):
        current_ei = offspring_ei[offspring_i]

        current_ey = None
        if offspring_ey is not None:
            current_ey = offspring_ey[offspring_i]

        replacement_i = None
        while rr_index_queue:
            replacement_i = rr_index_queue.pop()

            # Simple Metropolis sampling:
            # Always accept replacement if replacement is better (ratio > 1).
            # Otherwise, accept with probability equal to the ratio.
            if rr_ei[replacement_i] / current_ei > rng.rand():
                break

            # Hedge against greedy EI by Metropolis-sampling on the prediction.
            # Always accept if replacement is twice as good.
            if current_ey is not None:
                if current_ey / rr_ey[replacement_i] / 2 > rng.rand():
                    break

            replacement_i = None  # reset if failed

        if replacement_i is not None:
            yield rr_samples[replacement_i]
        else:
            yield offspring_samples[offspring_i]


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
    surrogate_model_class=SurrogateModelGPR,
    surrogate_model_args: dict=dict(),
) -> OptimizationResult:

    if logger is None:
        logger = Logger()

    assert popsize < max_nevals

    start_time = time.time()

    population = [
        Individual(space.sample(rng=rng), None)
        for _ in range(popsize)
    ]

    all_evaluations = []
    all_models = []

    population_fitness = await asyncio.gather(*(
        objective(ind.sample, fork_random_state(rng))
        for ind in population))
    for ind, fitness in zip(population, population_fitness):
        ind.fitness = fitness
        all_evaluations.append(ind)
    logger.record_evaluations(population, space=space)

    model = surrogate_model_class.estimate(
        [ind.sample for ind in all_evaluations],
        [ind.fitness for ind in all_evaluations],
        space=space, rng=rng, **surrogate_model_args)
    all_models.append(model)

    generation = 0
    fmin = min(ind.fitness for ind in all_evaluations)
    while len(all_evaluations) < max_nevals:
        generation += 1
        relscale = relscale_initial * (relscale_attenuation**(generation - 1))

        logger.announce_new_generation(
            generation, model=model, relscale=relscale)

        # generate new individuals
        offspring_samples: t.List[list] = \
            [ind.sample for ind in population]
        offspring_ei: t.List[float] = []
        candidate_chain_length = 1
        assert candidate_chain_length > 0
        for i in range(candidate_chain_length):
            offspring_samples, offspring_ei = suggest_next_candidate_a(
                offspring_samples,
                relscale=relscale * (relscale_attenuation ** i),
                fmin=fmin,
                rng=rng,
                breadth=20,
                model=model,
                space=space,
            )

        # Guard against premature convergence
        # by replacing worst estimated offspring with random individuals.
        # This is not completely random, but controlled by Metropolis sampling.
        offspring = [
            Individual(sample, None)
            for sample in improve_through_random_replacement(
                offspring_samples,
                offspring_ei=offspring_ei,
                offspring_ey=model.predict_a(
                    offspring_samples, return_std=False),
                model=model, space=space, fmin=fmin, rng=rng,
                n_replacements=popsize)
        ]

        # evaluate new individuals
        offspring_fitness = await asyncio.gather(*(
            objective(ind.sample, fork_random_state(rng))
            for ind in offspring))
        for ind, fitness in zip(offspring, offspring_fitness):
            ind.fitness = fitness
            all_evaluations.append(ind)
        logger.record_evaluations(offspring, space=space)

        # fit next model
        model = surrogate_model_class.estimate(
            [ind.sample for ind in all_evaluations],
            [ind.fitness for ind in all_evaluations],
            space=space, rng=rng, **surrogate_model_args)
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

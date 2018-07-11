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
            header=['utility', 'prediction', 'ei', 'cost', *param_names],
            formats=['{:.2f}', '{:.2f}', '{:.2e}', '{:.2f}', *param_formats],
            data=[[ind.fitness, ind.prediction, ind.ei, ind.cost, *ind.sample]
                  for ind in individuals],
        ))

    def announce_new_generation(
        self, gen: int, *,
        model: SurrogateModel,
        relscale: float,
    ) -> None:
        print(f"[INFO] starting generation #{gen}")
        print(f"       relscale {relscale:.5f}")
        print(f"       estimator: {model!r}")


class Individual(object):
    def __init__(
        self, sample: list, *,
        fitness: float = None,
        gen: int = None,
        ei: float = None,
        prediction: float = None,
        cost: float = None,
    ) -> None:

        self._sample = sample

        self._fitness = fitness
        self._gen = gen
        self._ei = ei
        self._prediction = prediction
        self._cost = cost

    def __repr__(self):
        fitness = self._fitness
        sample = ' '.join(repr(x) for x in self._sample)
        prediction = self._prediction
        ei = self._ei
        gen = self._gen
        cost = self._cost
        return (f'Individual({fitness} @{cost:.2f} [{sample}]'
                f' prediction: {prediction}'
                f' ei: {ei}'
                f' gen: {gen})')

    @property
    def sample(self) -> list:
        return self._sample

    @property
    def fitness(self) -> float:
        assert self._fitness is not None
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        assert self._fitness is None
        self._fitness = value

    @property
    def cost(self) -> float:
        assert self._cost is not None
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        assert self._cost is None
        self._cost = value

    @property
    def gen(self) -> int:
        assert self._gen is not None
        return self._gen

    @gen.setter
    def gen(self, value: int) -> None:
        assert self._gen is None
        self._gen = value

    @property
    def ei(self) -> float:
        assert self._ei is not None
        return self._ei

    @ei.setter
    def ei(self, value: float) -> None:
        assert self._ei is None
        self._ei = value

    @property
    def prediction(self) -> float:
        assert self._prediction is not None
        return self._prediction

    @prediction.setter
    def prediction(self, value: float) -> None:
        assert self._prediction is None
        self._prediction = value

    def is_fully_initialized(self) -> bool:
        return all(field is not None for field in (
            self._fitness, self._gen, self._ei, self._prediction, self._cost))


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
        assert all(ind.is_fully_initialized() for ind in all_individuals)

        self.all_individuals = all_individuals
        self.best_individual = min(
            all_individuals, key=lambda ind: ind.fitness)
        self.all_models = all_models
        self.duration = duration

    def best_n(self, n: int) -> t.List[Individual]:
        return sorted(self.all_individuals, key=lambda ind: ind.fitness)[:n]


def suggest_next_candidate(
    parent: Individual, *,
    relscale: float, fmin: float, breadth: int,
    rng: RandomState, model: SurrogateModel, space: Space,
) -> Individual:
    candidate_samples = [
        space.mutate(parent.sample, relscale=relscale, rng=rng)
        for _ in range(breadth)]
    candidate_mean, candidate_std = model.predict_a(candidate_samples)
    candidate_ei = expected_improvement(candidate_mean, candidate_std, fmin)
    i = np.argmax(candidate_ei)
    return Individual(
        candidate_samples[i],
        prediction=candidate_mean[i],
        ei=candidate_ei[i])


def suggest_next_candidate_a(
    parents: t.List[Individual], *,
    relscale: float, fmin: float, breadth: int,
    rng: RandomState, model: SurrogateModel, space: Space,
) -> t.List[Individual]:
    return [
        suggest_next_candidate(
            parent,
            relscale=relscale, fmin=fmin, rng=rng,
            breadth=breadth, model=model, space=space)
        for parent in parents]


def improve_through_random_replacement(
    offspring: t.List[Individual],
    *,
    hedge_via_prediction: bool = True,
    model: SurrogateModel,
    space: Space,
    fmin: float,
    rng: RandomState,
    n_replacements: int,
    n_suggestion_iters: int = 3,
    n_suggestion_breadth: int = 5,
    relscale_initial: float = 0.1,
    relscale_attenuation: float = 0.5,
) -> t.Iterable[Individual]:

    replacements = [
        Individual(space.sample(rng=rng)) for _ in range(n_replacements)]

    assert n_suggestion_iters > 0
    for i in range(n_suggestion_iters):
        relscale = relscale_initial * relscale_attenuation**i
        replacements = suggest_next_candidate_a(
            replacements,
            relscale=relscale, fmin=fmin,
            breadth=n_suggestion_breadth,
            rng=rng, model=model, space=space)

    # indices will be popped from the queue worst to best
    replacements = sorted(replacements, key=lambda ind: ind.ei, reverse=True)

    def metropolis_select(rng, ratio):
        return ratio > rng.rand()

    # search offspring from worst to best
    for current in sorted(offspring, key=lambda ind: ind.ei):
        replacement = None
        while replacements:
            replacement = replacements.pop()

            # Simple Metropolis sampling:
            # Always accept replacement if replacement is better (ratio > 1).
            # Otherwise, accept with probability equal to the ratio.
            if metropolis_select(rng, replacement.ei / current.ei):
                break

            # Hedge against greedy EI by Metropolis-sampling on the prediction.
            # Always accept if replacement is twice as good.
            if hedge_via_prediction and metropolis_select(
                rng, current.prediction / replacement.prediction / 2,
            ):
                break

            replacement = None  # reset if failed

        if replacement is not None:
            yield replacement
        else:
            yield current


Sample = list
ObjectiveFunction = t.Callable[
    [Sample, RandomState], t.Awaitable[t.Tuple[float, float]],
]


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
    surrogate_model_class: t.Type[SurrogateModel] =SurrogateModelGPR,
    surrogate_model_args: dict=dict(),
) -> OptimizationResult:

    if logger is None:
        logger = Logger()

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
        relscale = relscale_initial * (relscale_attenuation**(generation - 1))

        logger.announce_new_generation(
            generation, model=model, relscale=relscale)

        # generate new individuals
        offspring = population
        candidate_chain_length = 1
        assert candidate_chain_length > 0
        for i in range(candidate_chain_length):
            offspring = suggest_next_candidate_a(
                offspring,
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
        offspring = list(improve_through_random_replacement(
            offspring,
            model=model, space=space, fmin=fmin, rng=rng,
            n_replacements=popsize))

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

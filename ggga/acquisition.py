import abc
import typing as t
from .surrogate_model import SurrogateModel
from .util import minimize_by_gradient, TNumpy
from .individual import Individual
from .space import Space
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import attr
import scipy.stats  # type: ignore

Sample = list
IterableIndividuals = t.Iterable[Individual]


class AcquisitionStrategy(abc.ABC):
    @abc.abstractmethod
    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
    ) -> t.Iterator[Individual]:
        pass


class ChainedAcquisition(AcquisitionStrategy):
    def __init__(self, *strategies: AcquisitionStrategy) -> None:
        self.strategies: t.Collection[AcquisitionStrategy] = strategies

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
    ):
        offspring = population
        for strategy in self.strategies:
            offspring = strategy.acquire(
                offspring, model=model, relscale=relscale, rng=rng, fmin=fmin)
        return offspring


class HedgedAcquisition(AcquisitionStrategy):
    def __init__(self, *strategies: AcquisitionStrategy) -> None:
        self.strategies: t.Collection[AcquisitionStrategy] = strategies

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
    ):
        buckets: t.List[t.List[Individual]] = [[] for _ in self.strategies]
        for parent in population:
            buckets[rng.randint(len(buckets))].append(parent)
        for bucket, strategy in zip(buckets, self.strategies):
            yield from strategy.acquire(
                bucket, model=model, relscale=relscale, rng=rng, fmin=fmin)


@attr.s
class RandomReplacementAcquisition(AcquisitionStrategy):
    """Randomly replace bad individuals.

    This is not suitable as a primary acquisition strategy.

    Can be used to guard against premature convergence
    by replacing worst estimated offspring with random individuals.
    This is not completely random, but is controlled by Metropolis sampling.
    """

    n_replacements: int = attr.ib()
    space: Space = attr.ib()
    hedge_via_prediction: bool = True
    relscale_initial: float = 0.1
    subacquisition: AcquisitionStrategy = attr.ib()

    @subacquisition.default
    def _subacquisition_default(self):
        return RandomWalkAcquisition(
            breadth=5,
            candidate_chain_length=3,
            relscale_attenuation=0.5,
            space=self.space)

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
    ):
        relscale = np.clip(relscale, None, self.relscale_initial)

        replacements = [
            Individual(self.space.sample(rng=rng))
            for _ in range(self.n_replacements)]

        replacements = list(self.subacquisition.acquire(
            replacements, model=model, relscale=relscale, rng=rng, fmin=fmin))

        # indices will be popped from the queue worst to best
        replacements = sorted(
            replacements, key=lambda ind: ind.ei, reverse=True)

        def metropolis_select(rng, ratio):
            return ratio > rng.rand()

        # search offspring from worst to best
        for current in sorted(population, key=lambda ind: ind.ei):
            replacement = None
            while replacements:
                replacement = replacements.pop()

                # Simple Metropolis sampling:
                # Always accept replacement
                # if replacement is better (ratio > 1).
                # Otherwise, accept with probability equal to the ratio.
                if metropolis_select(rng, replacement.ei / current.ei):
                    break

                # Hedge against greedy EI
                # by Metropolis-sampling on the prediction.
                # Always accept if replacement is twice as good.
                if self.hedge_via_prediction and metropolis_select(
                    rng, current.prediction / replacement.prediction / 2,
                ):
                    break

                replacement = None  # reset if failed

            if replacement is not None:
                yield replacement
            else:
                yield current


@attr.s
class RandomWalkAcquisition(AcquisitionStrategy):
    breadth: int = attr.ib()
    candidate_chain_length: int = attr.ib()
    relscale_attenuation: float = attr.ib()
    space: Space = attr.ib()

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
    ):
        offspring = population

        assert self.candidate_chain_length > 0
        for i in range(self.candidate_chain_length):
            offspring = [
                self._one_step(
                    parent,
                    relscale=relscale * (self.relscale_attenuation ** i),
                    fmin=fmin,
                    rng=rng,
                    model=model,
                )
                for parent in offspring
            ]

        return offspring

    def _one_step(
        self, parent: Individual, *,
        relscale: np.ndarray,
        fmin: float,
        rng: RandomState,
        model: SurrogateModel,
    ) -> Individual:
        parent_sample_transformed = self.space.into_transformed(parent.sample)

        candidate_samples = [
            self.space.mutate_transformed(
                parent_sample_transformed, relscale=relscale, rng=rng)
            for _ in range(self.breadth)]
        for sample in candidate_samples:
            assert self.space.is_valid_transformed(sample), \
                f'mutated transformed sample must be valid: {sample!r}'

        candidate_mean, candidate_std = model.predict_transformed_a(
            candidate_samples)
        candidate_ei = expected_improvement(
            candidate_mean, candidate_std, fmin)

        i = np.argmax(candidate_ei)
        return Individual(
            self.space.from_transformed(candidate_samples[i]),
            prediction=candidate_mean[i],
            ei=candidate_ei[i])


@attr.s
class GradientAcquisition(AcquisitionStrategy):
    breadth: int = attr.ib()
    space: Space = attr.ib()
    # jitter_factor: float = 1/20

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel, relscale: np.ndarray, rng: RandomState,
        fmin: float,
    ):
        for parent in population:
            ind = self._one_step(
                parent, model=model, relscale=relscale, rng=rng, fmin=fmin)
            yield ind

    def _one_step(
        self, parent: Individual, *,
        model: SurrogateModel, relscale: np.ndarray, rng: RandomState,
        fmin: float,
    ):
        parent_sample_transformed = self.space.into_transformed(
            parent.sample)

        def objective(sample_transformed: Sample) -> float:
            mean, std = model.predict_transformed_a(sample_transformed)
            ei = expected_improvement(mean[0], std[0], fmin)
            return -ei

        def minimize_neg_ei(
            sample_transformed: Sample,
        ) -> t.Tuple[Sample, float]:
            opt_sample, opt_fitness = minimize_by_gradient(
                objective, sample_transformed,
                approx_grad=True,
                bounds=[p.transformed_bounds() for p in self.space.params],
            )
            assert self.space.is_valid_transformed(opt_sample), \
                f"optimized sample (transformed) not valid: {opt_sample}"
            return opt_sample, opt_fitness

        optimal_sample_transformed, optimal_fitness = minimize_neg_ei(
            parent_sample_transformed)

        for _ in range(self.breadth):
            starting_point = self.space.mutate_transformed(
                parent_sample_transformed, relscale=relscale, rng=rng)

            candidate_sample_transformed, candidate_fitness = \
                minimize_neg_ei(starting_point)

            if candidate_fitness < optimal_fitness:
                optimal_sample_transformed = candidate_sample_transformed
                optimal_fitness = candidate_fitness

        # # add a bit of jitter
        # optimal_sample_transformed = self.space.mutate_transformed(
        #     optimal_sample_transformed,
        #     relscale=relscale * self.jitter_factor,
        #     rng=rng,
        # )

        optimal_mean, optimal_std = model.predict_transformed_a(
            optimal_sample_transformed)

        return Individual(
            self.space.from_transformed(optimal_sample_transformed),
            prediction=optimal_mean[0],
            ei=expected_improvement(optimal_mean, optimal_std, fmin)[0])


def expected_improvement(mean: TNumpy, std: TNumpy, fmin: float) -> TNumpy:
    norm = scipy.stats.norm
    z = -(mean - fmin) / std
    ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z)
    return ei

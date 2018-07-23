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


class AcquisitionStrategy(abc.ABC):
    @abc.abstractmethod
    def acquire(
        self, population: t.Iterable[Individual], *,
        model: SurrogateModel,
        relscale: float,
        rng: RandomState,
        fmin: float,
    ) -> t.Iterator[Individual]:
        pass


class ChainedAcquisition(AcquisitionStrategy):
    def __init__(self, *strategies: AcquisitionStrategy) -> None:
        self.strategies: t.Collection[AcquisitionStrategy] = strategies

    def acquire(
        self, population: t.Iterable[Individual], *,
        model: SurrogateModel, relscale: float, rng: RandomState, fmin: float,
    ):
        offspring = population
        for strategy in self.strategies:
            offspring = strategy.acquire(
                offspring, model=model, relscale=relscale, rng=rng, fmin=fmin)
        return offspring


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
        self, population: t.Iterable[Individual], *,
        model: SurrogateModel, relscale: float, rng: RandomState,
        fmin: float,
    ):
        if self.relscale_initial < relscale:
            relscale = self.relscale_initial

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
        self, population: t.Iterable[Individual], *,
        model: SurrogateModel, relscale: float, rng: RandomState, fmin: float,
    ):
        offspring = population

        assert self.candidate_chain_length > 0
        for i in range(self.candidate_chain_length):
            offspring = (
                self._one_step(
                    parent,
                    relscale=relscale * (self.relscale_attenuation ** i),
                    fmin=fmin,
                    rng=rng,
                    model=model
                )
                for parent in offspring
            )

        return offspring

    def _one_step(
        self, parent: Individual, *,
        relscale: float, fmin: float, rng: RandomState, model: SurrogateModel,
    ) -> Individual:
        parent_sample_transformed = self.space.into_transformed(parent.sample)
        candidate_samples = [
            self.space.mutate_transformed(
                parent_sample_transformed, relscale=relscale, rng=rng)
            for _ in range(self.breadth)]
        candidate_mean, candidate_std = model.predict_a(candidate_samples)
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

    def acquire(
        self, population: t.Iterable[Individual], *,
        model: SurrogateModel, relscale: float, rng: RandomState,
        fmin: float,
    ):
        for parent in population:
            yield self._one_step(
                parent, model=model, relscale=relscale, rng=rng, fmin=fmin)

    def _one_step(
        self, parent: Individual, *,
        model: SurrogateModel, relscale: float, rng: RandomState,
        fmin: float,
    ):
        parent_sample_transformed = self.space.into_transformed(
            parent.sample)

        def objective_neg_ei(sample_transformed: Sample) -> float:
            mean, std = model.predict_transformed_a(sample_transformed)
            ei = expected_improvement(mean[0], std[0], fmin)
            return -ei

        def minimize_neg_ei(
            sample_transformed: Sample,
        ) -> t.Tuple[Sample, float]:
            opt_sample, opt_neg_ei = minimize_by_gradient(
                objective_neg_ei, sample_transformed,
                approx_grad=True,
                bounds=[p.transformed_bounds() for p in self.space.params],
            )
            assert self.space.is_valid_transformed(opt_sample), \
                f"optimized sample (transformed) not valid: {opt_sample}"
            return opt_sample, opt_neg_ei

        optimal_sample_transformed, optimal_neg_ei = minimize_neg_ei(
            parent_sample_transformed)

        for _ in range(self.breadth):
            starting_point = self.space.mutate_transformed(
                parent_sample_transformed, relscale=relscale, rng=rng)

            candidate_sample_transformed, candidate_neg_ei = \
                minimize_neg_ei(starting_point)

            if candidate_neg_ei < optimal_neg_ei:
                optimal_sample_transformed = candidate_sample_transformed
                optimal_neg_ei = candidate_neg_ei

        optimal_mean = model.predict_transformed_a(
            optimal_sample_transformed, return_std=False)

        return Individual(
            self.space.from_transformed(optimal_sample_transformed),
            prediction=optimal_mean[0],
            ei=-optimal_neg_ei)


def expected_improvement(mean: TNumpy, std: TNumpy, fmin: float) -> TNumpy:
    norm = scipy.stats.norm
    z = -(mean - fmin) / std
    ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z)
    return ei

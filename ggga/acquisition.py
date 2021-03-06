import abc
import typing as t

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import attr
import scipy.stats  # type: ignore

from .surrogate_model import SurrogateModel
from .util import minimize_by_gradient, TNumpy, yaml_constructor, Validator
from .individual import Individual
from .space import Space

Sample = list
IterableIndividuals = t.Iterable[Individual]


class AcquisitionStrategy(abc.ABC):
    """A strategy to acquire new samples.

    Implementations:

    .. autosummary::

        ChainedAcquisition
        HedgedAcquisition
        RandomReplacementAcquisition
        MutationAcquisition
        RandomWalkAcquisition
        GradientAcquisition
    """
    @abc.abstractmethod
    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ) -> t.Iterator[Individual]:
        """Acquire new individuals.

        Arguments
        ---------
        population
            Previous population/parents.
        model
            Current model of the utility landscape.
        relscale
            suggested normalized standard deviation for mutating individuals.
        fmin
            Current best observed value (useful for EI)
        space
        rng

        Returns
        -------
        typing.Iterable[Individual]
            A finite sequence of individuals
            that may be selected for evaluation.
        """


class ChainedAcquisition(AcquisitionStrategy):
    """Perform multi-stage acquisition.

    Each stage operates on the results of the previous stage.
    """

    def __init__(self, *strategies: AcquisitionStrategy) -> None:
        self.strategies: t.Iterable[AcquisitionStrategy] = strategies

        if not strategies:
            raise TypeError("at least one strategy required")

        for substrategy in strategies:
            if not isinstance(substrategy, AcquisitionStrategy):
                raise TypeError(f"not an AcquisitionStrategy: {substrategy!r}")

    @staticmethod
    @yaml_constructor('!ChainedAcquisition', safe=True)
    def from_yaml(loader, node) -> 'ChainedAcquisition':
        return ChainedAcquisition(*loader.construct_sequence(node))

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ):
        offspring = population
        for strategy in self.strategies:
            offspring = strategy.acquire(
                offspring, model=model, space=space,
                relscale=relscale, rng=rng, fmin=fmin)
        return offspring


class HedgedAcquisition(AcquisitionStrategy):
    """Randomly assign parent individuals to a sub-strategy."""

    def __init__(self, *strategies: AcquisitionStrategy) -> None:
        self.strategies: t.Iterable[AcquisitionStrategy] = strategies

        if not strategies:
            raise TypeError("at least one strategy required")

        for substrategy in strategies:
            if not isinstance(substrategy, AcquisitionStrategy):
                raise TypeError("not an AcquisitionStrategy: {substrategy!r}")

    @staticmethod
    @yaml_constructor('!HedgedAcquisition', safe=True)
    def from_yaml(loader, node) -> 'HedgedAcquisition':
        return HedgedAcquisition(*loader.load_sequence(node))

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ):
        buckets: t.List[t.List[Individual]] = [[] for _ in self.strategies]
        for parent in population:
            buckets[rng.randint(len(buckets))].append(parent)
        for bucket, strategy in zip(buckets, self.strategies):
            yield from strategy.acquire(
                bucket, model=model, space=space,
                relscale=relscale, rng=rng, fmin=fmin)


@attr.s
class RandomReplacementAcquisition(AcquisitionStrategy):
    """Replace bad individuals with random samples.

    This is not suitable as a primary acquisition strategy.

    Can be used to guard against premature convergence
    by replacing worst estimated offspring with random individuals.
    This is not completely random, but is controlled by Metropolis sampling.
    """

    n_replacements: int = attr.ib()
    n_replacements.validator(Validator.is_posint)  # type: ignore

    hedge_via_prediction: bool = attr.ib(default=True)
    hedge_via_prediction.validator(Validator.is_instance())  # type: ignore

    relscale_initial: float = attr.ib(default=0.1)
    relscale_initial.validator(Validator.is_percentage)  # type: ignore

    subacquisition: AcquisitionStrategy = attr.ib()
    subacquisition.validator(Validator.is_instance())  # type: ignore

    @staticmethod
    @yaml_constructor('!RandomReplacementAcquisition', safe=True)
    def from_yaml(loader, node) -> 'RandomReplacementAcquisition':
        if node.id == 'scalar':
            return RandomReplacementAcquisition(
                n_replacements=loader.construct_scalar(node))
        else:
            return RandomReplacementAcquisition(
                **loader.construct_mapping(node))

    @subacquisition.default
    def _subacquisition_default(self):
        return RandomWalkAcquisition(
            breadth=5,
            candidate_chain_length=3,
            relscale_attenuation=0.5)

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ):
        relscale = np.clip(relscale, None, self.relscale_initial)

        replacements = [
            Individual(space.sample(rng=rng))
            for _ in range(self.n_replacements)]

        replacements = list(self.subacquisition.acquire(
            replacements, model=model, space=space,
            relscale=relscale, rng=rng, fmin=fmin))

        def key_expected_improvement(ind: Individual) -> float:
            return ind.expected_improvement

        # indices will be popped from the queue worst to best
        replacements = sorted(
            replacements, key=key_expected_improvement, reverse=True)

        # search offspring from worst to best
        for current in sorted(population, key=key_expected_improvement):
            replacement = None
            while replacements:
                replacement = replacements.pop()

                # Select replacement via Metropolis Sampling on EI
                repl_ei = replacement.expected_improvement
                curr_ei = current.expected_improvement
                if metropolis_select(rng, repl_ei, curr_ei):
                    break

                # So EI didn't introduce a replacement.
                # But what if the replacement is much better?
                # This should not matter, as the prediction value
                # is already considered by the EI metric.
                # In practice, being willing to do a bit more exploration
                # can be helpful to get a good model.
                #
                # A previous version tried to metropolis-select directly
                # on the prediction value, but the prediction is
                # interval-scaled, NOT ratio-scaled!
                #
                # We can introduce a pseudo-interval-scale by selecting
                # on the difference to some zero point. Here we can choose
                # ratio = exp(curr_prediction - repl_prediction).
                # - if the replacement is better (lower) the metropolis ratio
                #   will be > 1 and the replacement will be selected.
                # - if the replacement is worse (higher)
                #   the replacement will be selected with a smallish chance.
                # Note that for predictions, smaller is better.
                if self.hedge_via_prediction:
                    curr_prediction = current.prediction
                    repl_prediction = replacement.prediction
                    ratio = np.exp(curr_prediction - repl_prediction)
                    if metropolis_select(rng, ratio, 1.0):
                        break

                replacement = None  # reset if failed

            if replacement is not None:
                yield replacement
            else:
                yield current


@attr.s
class MutationAcquisition(AcquisitionStrategy):
    """Randomly mutate each parent to create new samples in their neighborhood.
    """

    breadth: int = attr.ib()
    breadth.validator(Validator.is_posint)  # type: ignore

    @staticmethod
    @yaml_constructor('!MutationAcquisition', safe=True)
    def from_yaml(loader, node) -> 'MutationAcquisition':
        return MutationAcquisition(**loader.construct_mapping(node))

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ):
        for parent in population:
            parent_sample_transformed = space.into_transformed(parent.sample)

            candidate_samples = [
                space.mutate_transformed(
                    parent_sample_transformed, relscale=relscale, rng=rng)
                for _ in range(self.breadth)]

            candidate_mean, candidate_std = model.predict_transformed_a(
                candidate_samples)
            candidate_ei = expected_improvement(
                candidate_mean, candidate_std, fmin)

            i = np.argmax(candidate_ei)
            yield Individual(
                space.from_transformed(candidate_samples[i]),
                prediction=candidate_mean[i],
                expected_improvement=candidate_ei[i])


@attr.s
class RandomWalkAcquisition(AcquisitionStrategy):
    breadth: int = attr.ib()
    breadth.validator(Validator.is_posint)  # type: ignore

    steps: int = attr.ib()
    steps.validator(Validator.is_posint)  # type: ignore

    relscale_attenuation: float = attr.ib(default=0.5)
    relscale_attenuation.validator(Validator.is_percentage)  # type: ignore

    @staticmethod
    @yaml_constructor('!RandomWalkAcquisition', safe=True)
    def from_yaml(loader, node) -> 'RandomWalkAcquisition':
        return RandomWalkAcquisition(**loader.construct_mapping(node))

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel,
        relscale: np.ndarray,
        rng: RandomState,
        fmin: float,
        space: Space,
    ):
        subacq = MutationAcquisition(breadth=self.breadth)

        offspring = population

        assert self.steps > 0
        for i in range(self.steps):
            offspring = list(subacq.acquire(
                offspring,
                relscale=relscale * (self.relscale_attenuation ** i),
                fmin=fmin,
                rng=rng,
                model=model,
                space=space,
            ))

        return offspring


@attr.s
class GradientAcquisition(AcquisitionStrategy):
    """Use gradient optimization to find optimal samples."""

    breadth: int = attr.ib()
    breadth.validator(Validator.is_posint)  # type: ignore

    # jitter_factor: float = 1/20

    @staticmethod
    @yaml_constructor('!GradientAcquisition', safe=True)
    def from_yaml(loader, node) -> 'GradientAcquisition':
        if node.id == 'scalar':
            return GradientAcquisition(breadth=loader.construct_scalar(node))
        else:
            return GradientAcquisition(**loader.construct_mapping(node))

    def acquire(
        self, population: IterableIndividuals, *,
        model: SurrogateModel, relscale: np.ndarray, rng: RandomState,
        fmin: float,
        space: Space,
    ):
        def suggest_neighbor(transformed: Sample) -> Sample:
            return space.mutate_transformed(
                transformed, relscale=relscale, rng=rng)

        def objective(transformed: Sample) -> float:
            mean, std = model.predict_transformed_a(transformed)
            assert std is not None
            return -expected_improvement(mean[0], std[0], fmin)

        def optimize_via_gradient(
            transformed: Sample
        ) -> t.Tuple[Sample, float]:
            opt_sample, opt_fitness = minimize_by_gradient(
                objective, transformed,
                approx_grad=True,
                bounds=[p.transformed_bounds() for p in space.params],
            )
            assert space.is_valid_transformed(opt_sample), \
                f"optimized sample (transformed) not valid: {opt_sample}"
            return opt_sample, opt_fitness

        for parent in population:
            sample = self._optimize_sample_with_restart(
                space.into_transformed(parent.sample),
                suggest_neighbor=suggest_neighbor,
                optimize_sample=optimize_via_gradient,
            )

            vec_mean, vec_std = model.predict_transformed_a([sample])
            assert vec_std is not None

            yield Individual(
                space.from_transformed(sample),
                prediction=vec_mean[0],
                expected_improvement=expected_improvement(
                    vec_mean[0], vec_std[0], fmin),
            )

    def _optimize_sample_with_restart(
        self, parent_sample_transformed: Sample, *,
        suggest_neighbor: t.Callable[[Sample], Sample],
        optimize_sample: t.Callable[[Sample], t.Tuple[Sample, float]],
    ):
        optimal_sample_transformed, optimal_fitness = optimize_sample(
            parent_sample_transformed)

        for _ in range(self.breadth):
            candidate_sample_transformed, candidate_fitness = \
                optimize_sample(
                    suggest_neighbor(
                        parent_sample_transformed))

            if candidate_fitness < optimal_fitness:
                optimal_sample_transformed = candidate_sample_transformed
                optimal_fitness = candidate_fitness

        return optimal_sample_transformed


def expected_improvement(
    vec_mean: TNumpy, vec_std: TNumpy, fmin: float,
) -> TNumpy:
    norm = scipy.stats.norm
    vec_z = -(vec_mean - fmin) / vec_std
    vec_ei = -(vec_mean - fmin) * norm.cdf(vec_z) + vec_std * norm.pdf(vec_z)
    return vec_ei


def metropolis_select(
    rng: RandomState, option: float, benchmark: float,
) -> bool:
    r"""Decide whether an option should be chosen:

    - Always select the option if it improves over the benchmark.
    - Select randomly with probability p=option/benchmark otherwise.

    Corner case: if option = benchmark = 0, select with p=50%.

    Better = larger. Only applicable to ratio-scaled variables!
    """
    assert option >= 0
    assert benchmark >= 0
    if option > benchmark:
        return True
    if benchmark == 0:
        ratio = 0.5
    else:
        ratio = option/benchmark

    return ratio > rng.rand()

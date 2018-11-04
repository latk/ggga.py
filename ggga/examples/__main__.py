import abc
import argparse
import asyncio
import typing as t

import attr
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from .. import Space, Real, Minimizer, ObjectiveFunction, RandomState
from .. import SurrogateModel, SurrogateModelGPR, SurrogateModelKNN
from ..benchmark_functions import (
    goldstein_price, easom, himmelblau, rastrigin, rosenbrock, sphere)
from ..outputs import Output
from ..visualization import PartialDependence

StrategyResult = t.Tuple[
    SurrogateModel, np.ndarray, np.ndarray, t.Optional[float]]


@attr.s
class Example:
    function: t.Callable[..., float] = attr.ib()
    space: Space = attr.ib()
    minima: t.List[t.Tuple[list, float]] = attr.ib()

    def make_objective(self, *, log_y: bool) -> ObjectiveFunction:

        async def objective(xs, _rng):
            y = self.function(*xs)

            if log_y:
                assert y > 0, f"ys must be positive, was {y}"
                y = np.log(y)

            cost = 0.0
            return y, cost

        return objective


EXAMPLES: t.Dict[str, Example] = {}

EXAMPLES['goldstein-price'] = Example(
    function=goldstein_price,
    space=Space(
        Real('x_1', -2, 2),
        Real('x_2', -2, 2),
    ),
    minima=[
        ([0.0, -1.0], 0.0),
    ],
)


# Easom is very difficult for the GPR
# because the initial random samples will likely miss the minimum.
# Given that all observations are flat, the GPR will choose a long scale
# which assigns a low EI to unexplored regions.
EXAMPLES['easom'] = Example(
    function=lambda x_1, x_2: easom(x_1, x_2, amplitude=100.0),
    space=Space(
        Real('x_1', -25, 25),
        Real('x_2', -25, 25),
    ),
    minima=[
        ([np.pi, np.pi], 0.0)
    ],
)

EXAMPLES['himmelblau'] = Example(
    function=himmelblau,
    space=Space(
        Real('x_1', -5, 5),
        Real('x_2', -5, 5),
    ),
    minima=[
        ([3.0, 2.0], 0.0),
        ([-2.805118, 3.131312], 0.0),
        ([-3.779310, -3.283186], 0.0),
        ([3.584428, -1.848126], 0.0),
    ],
)

EXAMPLES['rastrigin2'] = Example(
    function=rastrigin,
    space=Space(
        Real('x_1', -5.12, 5.12),
        Real('x_2', -5.12, 5.12),
    ),
    minima=[([0.0]*2, 0.0)],
)

EXAMPLES['rastrigin6'] = Example(
    function=rastrigin,
    space=Space(
        Real('x_1', -5.12, 5.12),
        Real('x_2', -5.12, 5.12),
        Real('x_3', -5.12, 5.12),
        Real('x_4', -5.12, 5.12),
        Real('x_5', -5.12, 5.12),
        Real('x_6', -5.12, 5.12),
    ),
    minima=[([0.0]*6, 0.0)],
)

EXAMPLES['rosenbrock2'] = Example(
    function=rosenbrock,
    space=Space(
        Real('x_1', -5.12, 5.12),
        Real('x_2', -5.12, 5.12),
    ),
    minima=[([1.0]*2, 0.0)],
)

EXAMPLES['rosenbrock6'] = Example(
    function=rosenbrock,
    space=Space(
        Real('x_1', -5.12, 5.12),
        Real('x_2', -5.12, 5.12),
        Real('x_3', -5.12, 5.12),
        Real('x_4', -5.12, 5.12),
        Real('x_5', -5.12, 5.12),
        Real('x_6', -5.12, 5.12),
    ),
    minima=[([1.0]*6, 0.0)],
)

EXAMPLES['sphere2'] = Example(
    function=sphere,
    space=Space(
        Real('x_1', -2.0, 2.0),
        Real('x_2', -2.0, 2.0),
    ),
    minima=[([0.0]*2, 0.0)],
)

EXAMPLES['sphere6'] = Example(
    function=sphere,
    space=Space(
        Real('x_1', -2.0, 2.0),
        Real('x_2', -2.0, 2.0),
        Real('x_3', -2.0, 2.0),
        Real('x_4', -2.0, 2.0),
        Real('x_5', -2.0, 2.0),
        Real('x_6', -2.0, 2.0),
    ),
    minima=[([0.0]*6, 0.0)],
)


@attr.s
class StrategyConfiguration:
    space: Space = attr.ib()
    n_samples: int = attr.ib()
    surrogate_model_class: t.Type[SurrogateModel] = attr.ib()
    quiet: bool = attr.ib()


class Strategy(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def run(
        self, objective: ObjectiveFunction, *,
        cfg: StrategyConfiguration,
        rng: RandomState,
    ) -> StrategyResult:
        raise NotImplementedError


class RandomStrategy(Strategy):
    name = 'random'

    @staticmethod
    async def run(
        objective: ObjectiveFunction, *,
        cfg: StrategyConfiguration,
        rng: RandomState,
    ) -> StrategyResult:
        # evaluate a couple of random samples
        xs = np.array([
            cfg.space.sample(rng=rng) for _ in range(cfg.n_samples)
        ])
        ys = np.array([
            y
            for x in xs
            for y, cost in [await objective(x, rng)]
        ])
        model = cfg.surrogate_model_class.estimate(
            xs, ys, space=cfg.space, rng=rng, prior=None)
        return model, xs, ys, None


class GGGAStrategy(Strategy):
    name = 'GGGA'

    @staticmethod
    async def run(
        objective: ObjectiveFunction, *,
        cfg: StrategyConfiguration,
        rng: RandomState,
    ) -> StrategyResult:
        minimizer = Minimizer(
            max_nevals=cfg.n_samples,
            surrogate_model_class=cfg.surrogate_model_class,
        )
        res = await minimizer.minimize(
            objective, space=cfg.space, rng=rng,
            outputs=(
                Output(space=cfg.space, log_file=None)
                if cfg.quiet else None),
        )
        return res.model, res.xs, res.ys, res.fmin

# TODO: Sobol strategy, Irace strategy


async def run_example_with_strategies(  # pylint: disable=too-many-locals
    example_name: str, example: Example, *,
    strategies: t.List[Strategy],
    cfg: StrategyConfiguration,
    rng_seed: int,
    log_y: bool,
) -> t.List[plt.Figure]:
    objective = example.make_objective(log_y=log_y)

    figs = []
    for strategy in strategies:
        rng = RandomState(rng_seed)

        model, xs, ys, fmin = await strategy.run(objective, rng=rng, cfg=cfg)

        compare_model_with_minima_io(
            model, example.minima,
            fmin=fmin,
            log_y=log_y,
            n_samples=cfg.n_samples,
            sample_type=strategy.name,
        )

        fig, _ = PartialDependence(model=model, space=example.space, rng=rng) \
            .plot_grid(xs, ys)
        figtitle = example_name.title()
        fig.suptitle(f"{figtitle} ({cfg.n_samples} {strategy.name} samples)")
        figs.append(fig)

    return figs


def compare_model_with_minima_io(
    model: SurrogateModel, minima: t.List[t.Tuple[list, float]], *,
    fmin: t.Optional[float],
    log_y: bool,
    n_samples: int,
    sample_type: str,
) -> None:

    if log_y:
        fmin = None if fmin is None else np.exp(fmin)

    fmin_str = ("(None)" if fmin is None else f"{fmin:.2f}")
    print(f"Minima after {n_samples} {sample_type} samples: {fmin_str}")
    for x_min, y_min_expected in minima:
        y_min, y_min_std = model.predict(x_min)
        assert y_min_std is not None
        if log_y:
            y_min = np.exp(y_min)
        x_min_str = ', '.join(f"{x:.2f}" for x in x_min)
        print(
            f"  * f({x_min_str}) = {y_min:.2f} ± {y_min_std:.2f} "
            f"expected {y_min_expected:.2f}")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.description = "Run an optimization benchmark function."

    parser.add_argument(
        'example', choices=sorted(EXAMPLES),
        help="The example function to run.")
    parser.add_argument(
        '--no-interactive',
        dest='interactive', action='store_false',
        help="Do not display the generated plots.")
    parser.add_argument(
        '--samples', metavar='N', type=int,
        dest='samples', default=50,
        help="How many evaluations should be sampled. Default: %(default)s.")
    parser.add_argument(
        '--logy', action='store_true',
        dest='log_y',
        help="Log-transform the objective function.")
    parser.add_argument(
        '--model', choices=('gpr', 'knn'),
        dest='model', default='gpr',
        help="The surrogate model implementation used for prediction. "
             "gpr: Gaussian Process regression. "
             "knn: k-Nearest Neighbor. "
             "Default: %(default)s.")
    parser.add_argument(
        '--quiet', action='store_true',
        dest='quiet',
        help="Don't display human-readable output during minimization.")
    parser.add_argument(
        '--seed', metavar='SEED', type=int,
        dest='seed', default=7861,
        help="Seed for reproducible runs. Default: %(default)s.")

    return parser


def main() -> None:
    options = make_argument_parser().parse_args()

    surrogate_model_class: t.Type[SurrogateModel]
    if options.model == 'gpr':
        surrogate_model_class = SurrogateModelGPR
    elif options.model == 'knn':
        surrogate_model_class = SurrogateModelKNN
    else:
        raise ValueError(f"Unknown argument value: --model {options.model}")

    example = EXAMPLES[options.example]

    strategy_cfg = StrategyConfiguration(
        space=example.space,
        n_samples=options.samples,
        surrogate_model_class=surrogate_model_class,
        quiet=options.quiet,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example_with_strategies(
        options.example, example,
        strategies=[RandomStrategy(), GGGAStrategy()],
        cfg=strategy_cfg,
        rng_seed=options.seed,
        log_y=options.log_y,
    ))

    if options.interactive:
        plt.show()


if __name__ == '__main__':
    main()

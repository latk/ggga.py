import argparse
import asyncio
import typing as t

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import yaml


from . import RandomState
from . import SurrogateModel, SurrogateModelGPR, SurrogateModelKNN
from .visualization import PartialDependence
from .util import yaml_constructor
from .outputs import RecordCompletedEvaluations
from .examples import (
    Example, EXAMPLES,
    Strategy, StrategyConfiguration,
    RandomStrategy, GGGAStrategy, IraceStrategy
)


@yaml_constructor('!Irace', safe=True)
def irace_from_yaml(loader, node) -> 'IraceStrategy':
    args = loader.construct_mapping(node)
    return IraceStrategy(**args)


@yaml_constructor('!GGGA', safe=True)
def ggga_from_yaml(loader, node) -> 'GGGAStrategy':
    args = loader.construct_mapping(node)
    return GGGAStrategy(minimizer_args=args)


async def run_example_with_strategies(  # pylint: disable=too-many-locals
    example_name: str, example: Example, *,
    strategies: t.List[Strategy],
    cfg: StrategyConfiguration,
    rng_seed: int,
    log_y: bool,
    noise_level: float,
    render_plots: bool,
) -> t.List[plt.Figure]:

    if cfg.csv_file:
        csv_writer = RecordCompletedEvaluations.new(
            cfg.csv_file, space=cfg.space)

        def on_evaluation(sample, value):
            csv_writer.write_result(sample=sample, observation=value)
    else:
        def on_evaluation(sample, value):  # pylint: disable=unused-argument
            pass

    objective = example.make_objective(
        log_y=log_y, noise_level=noise_level, on_evaluation=on_evaluation)

    figs = []
    for strategy in strategies:
        rng = RandomState(rng_seed)

        model, xs, ys, fmin, min_sample = \
            await strategy.run(objective, rng=rng, cfg=cfg)

        compare_model_with_minima_io(
            model, example.minima,
            fmin=fmin,
            min_sample=min_sample,
            log_y=log_y,
            n_samples=cfg.n_samples,
            sample_type=strategy.name,
        )

        if not render_plots:
            continue

        fig, _ = PartialDependence(model=model, space=example.space, rng=rng) \
            .plot_grid(xs, ys)
        figtitle = example_name.title()
        fig.suptitle(f"{figtitle} ({cfg.n_samples} {strategy.name} samples)")
        figs.append(fig)

    return figs


def compare_model_with_minima_io(
    model: SurrogateModel, minima: t.List[t.Tuple[list, float]], *,
    fmin: float,
    min_sample: list,
    log_y: bool,
    n_samples: int,
    sample_type: str,
) -> None:

    if log_y:
        fmin = np.exp(fmin)

    sample_str = ', '.join(
        param.fmt.format(x)
        for x, param in zip(min_sample, model.space.params))

    print(
        f"Minima after {n_samples} {sample_type} samples: "
        f"{fmin:.2f} @ ({sample_str})")
    for x_min, y_min_expected in minima:
        y_min, y_min_std = model.predict(x_min)
        assert y_min_std is not None
        if log_y:
            y_min = np.exp(y_min)
        x_min_str = ', '.join(f"{x:.2f}" for x in x_min)
        print(
            f"  * f({x_min_str}) = {y_min:.2f} ± {y_min_std:.2f} "
            f"expected {y_min_expected:.2f}")


def load_strategy(spec: str) -> Strategy:
    if spec.startswith('!'):
        try:
            strategy = yaml.safe_load(spec)
        except Exception as ex:
            raise argparse.ArgumentTypeError(
                f"YAML did not parse successfully: {ex}")
        if not isinstance(strategy, Strategy):
            raise argparse.ArgumentTypeError(
                f"YAML did not represent a strategy, "
                f"but <{type(strategy).__name__}>: {strategy!r}")
        return strategy

    strategies: t.List[t.Type[Strategy]] = [GGGAStrategy, RandomStrategy]
    for strategy_class in strategies:
        if spec.casefold() == t.cast(str, strategy_class.name).casefold():
            return strategy_class()

    raise argparse.ArgumentTypeError(f"Cannot find a strategy called {spec!r}")


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
        '--noise', metavar='NOISE_LEVEL', type=float,
        dest='noise', default=0.0,
        help="Standard deviation of test function noise. "
             "Default: %(default)s.")
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
    parser.add_argument(
        '-s', '--strategy', metavar='STRATEGY', type=load_strategy,
        dest='strategies', action='append',
        help="Which optimization strategies will be used. "
             "Can be 'random', 'ggga', "
             "or a YAML document describing the strategy.")
    parser.add_argument(
        '--csv', metavar='FILE', type=argparse.FileType('w'),
        dest='csv',
        help="Write evaluations results to a CSV file. "
             "Only use this when running a single strategy.")

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

    strategies = options.strategies
    if not strategies:
        strategies = [RandomStrategy(), GGGAStrategy()]

    example = EXAMPLES[options.example]

    if options.csv and len(strategies) > 1:
        raise ValueError("--csv can only be used with a single strategy")

    strategy_cfg = StrategyConfiguration(
        space=example.space,
        n_samples=options.samples,
        surrogate_model_class=surrogate_model_class,
        quiet=options.quiet,
        csv_file=options.csv,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example_with_strategies(
        options.example, example,
        strategies=strategies,
        cfg=strategy_cfg,
        rng_seed=options.seed,
        log_y=options.log_y,
        noise_level=options.noise,
        render_plots=options.interactive,
    ))

    if options.interactive:
        plt.show()


if __name__ == '__main__':
    main()

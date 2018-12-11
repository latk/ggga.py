import asyncio
import typing as t

import click
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import yaml


from . import RandomState
from . import SurrogateModel, SurrogateModelGPR, SurrogateModelKNN
from .visualization import PartialDependence, DualDependenceStyle
from .util import yaml_constructor
from .outputs import RecordCompletedEvaluations
from .examples import (
    Example, EXAMPLES,
    Strategy, StrategyConfiguration,
    RandomStrategy, GGGAStrategy, IraceStrategy
)


@yaml_constructor('!Irace', safe=True)
def irace_from_yaml(loader, node) -> IraceStrategy:
    args = loader.construct_mapping(node)
    return IraceStrategy(**args)


@yaml_constructor('!GGGA', safe=True)
def ggga_from_yaml(loader, node) -> GGGAStrategy:
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
    style: DualDependenceStyle = None,
) -> t.List[plt.Figure]:

    if cfg.csv_file:
        csv_writer = RecordCompletedEvaluations.new(
            cfg.csv_file, space=cfg.space)

        def on_evaluation(sample, value):
            csv_writer.write_result(sample=sample, observation=value)
    else:
        def on_evaluation(sample, value):  # pylint: disable=unused-argument
            pass

    if style is None:
        style = DualDependenceStyle()

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
            .plot_grid(xs, ys, style=style)
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


class StrategyParam(click.ParamType):
    name = 'strategy'

    def convert(self, spec: str, param, ctx):
        if spec.startswith('!'):
            try:
                strategy = yaml.safe_load(spec)
            except Exception as ex:
                self.fail(f"YAML did not parse successfully: {ex}", param, ctx)
            if not isinstance(strategy, Strategy):
                self.fail(
                    f"YAML did not represent a strategy, "
                    f"but <{type(strategy).__name__}>: {strategy!r}",
                    param, ctx)
            return strategy

        strategies: t.List[t.Type[Strategy]] = [GGGAStrategy, RandomStrategy]
        for strategy_class in strategies:
            if spec.casefold() == t.cast(str, strategy_class.name).casefold():
                return strategy_class()

        self.fail(f"Cannot find a strategy called {spec!r}", param, ctx)


@click.command()
@click.argument(
    'example_name', type=click.Choice(sorted(EXAMPLES)))
@click.option(
    '--interactive/--no-interactive', default=True,
    help="Whether to display the generated plots.")
@click.option(
    '--samples', type=int, metavar='N',
    default=50, show_default=True,
    help="How many evaluations should be sampled.")
@click.option(
    '--logy', is_flag=True, default=False,
    help="Log-transform the objective function.")
@click.option(
    '--noise', type=float, default=0.0, show_default=True,
    help="Standard deviation of test function noise.")
@click.option(
    '--model', type=click.Choice(['gpr', 'knn']),
    default='gpr', show_default=True,
    help="The surrogate model implementation used for prediction. "
         "gpr: Gaussian Process Regression. knn: k-Nearest Neighbor.")
@click.option(
    '--quiet', is_flag=True,
    help="Don't display human-readable output during minimization.")
@click.option(
    '--seed', metavar='SEED', type=int,
    default=7861, show_default=True,
    help="Seed for reproducible runs.")
@click.option(
    '-s', '--strategy', 'strategies', metavar='STRATEGY', type=StrategyParam(),
    multiple=True,
    help="Which optimization strategy will be used. "
         "Can be 'random', 'ggga', "
         "or a YAML document describing the strategy.")
@click.option(
    '--csv', metavar='FILE', type=click.File('w'),
    help="Write evaluation results to a CSV file. "
         "Only use this when running a single strategy.")
@click.option(
    '--style', metavar='STYLE',
    help="DualDependenceStyle for the plots.")
def cli(
    example_name, *,
    interactive, samples, logy, noise, model, quiet,
    seed, strategies, csv, style,
):
    """Run an EXAMPLE optimization benchmark function"""

    surrogate_model_class: t.Type[SurrogateModel]
    if model == 'gpr':
        surrogate_model_class = SurrogateModelGPR
    elif model == 'knn':
        surrogate_model_class = SurrogateModelKNN
    else:
        raise ValueError(f"Unknown argument value: --model {model}")

    if not strategies:
        strategies = [RandomStrategy(), GGGAStrategy()]

    example = EXAMPLES[example_name]

    if csv and len(strategies) > 1:
        raise ValueError("--csv can only be used with a single strategy")

    strategy_cfg = StrategyConfiguration(
        space=example.space,
        n_samples=samples,
        surrogate_model_class=surrogate_model_class,
        quiet=quiet,
        csv_file=csv,
    )

    style = None
    if style is not None:
        style = DualDependenceStyle(**yaml.safe_load(style))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example_with_strategies(
        example_name, example,
        strategies=strategies,
        cfg=strategy_cfg,
        rng_seed=seed,
        log_y=logy,
        noise_level=noise,
        render_plots=interactive,
        style=style
    ))

    if interactive:
        plt.show()


if __name__ == '__main__':
    cli()

import asyncio
import io
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
    EXAMPLES, Example, ExampleWithVariableDimensions,
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


def run_example(
    example_name: str, *,
    csv: io.StringIO,
    interactive: bool,
    logy: bool,
    model: str,
    noise: float,
    samples: int,
    seed: int,
    strategies: t.List[Strategy],
    style: str,
    quiet: bool,
    dimensions: int = None,
) -> None:

    surrogate_model_class: t.Type[SurrogateModel]
    if model == 'gpr':
        surrogate_model_class = SurrogateModelGPR
    elif model == 'knn':
        surrogate_model_class = SurrogateModelKNN
    else:
        raise ValueError(f"Unknown argument value: --model {model}")

    if not strategies:
        strategies = [RandomStrategy(), GGGAStrategy()]

    if csv and len(strategies) > 1:
        raise ValueError("--csv can only be used with a single strategy")

    if style is not None:
        ddstyle = DualDependenceStyle(**yaml.safe_load(style))
    else:
        ddstyle = None

    example = EXAMPLES[example_name].fix_dimension(dimensions)

    strategy_cfg = StrategyConfiguration(
        space=example.space,
        n_samples=samples,
        surrogate_model_class=surrogate_model_class,
        quiet=quiet,
        csv_file=csv,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example_with_strategies(
        example_name, example,
        strategies=strategies,
        cfg=strategy_cfg,
        rng_seed=seed,
        log_y=logy,
        noise_level=noise,
        render_plots=interactive,
        style=ddstyle
    ))

    if interactive:
        plt.show()


def click_common_example_options(provide_defaults: bool = True):

    def option(*args, default=None, show_default: bool = None, **kwargs):

        if not provide_defaults:
            show_default = False
            default = None

        return click.option(
                *args,
                default=default,
                show_default=(show_default or False),
                **kwargs)

    common_options = [
        option(
            '--interactive/--no-interactive', default=True,
            help="Whether to display the generated plots."),
        option(
            '--samples', type=int, metavar='N',
            default=50, show_default=True,
            help="How many evaluations should be sampled."),
        option(
            '--logy', is_flag=True, default=False,
            help="Log-transform the objective function."),
        option(
            '--noise', type=float, default=0.0, show_default=True,
            help="Standard deviation of test function noise."),
        option(
            '--model', type=click.Choice(['gpr', 'knn']),
            default='gpr', show_default=True,
            help="The surrogate model implementation used for prediction. "
                 "gpr: Gaussian Process Regression. knn: k-Nearest Neighbor."),
        option(
            '--quiet', is_flag=True,
            help="Don't display human-readable output during minimization."),
        option(
            '--seed', metavar='SEED', type=int,
            default=7861, show_default=True,
            help="Seed for reproducible runs."),
        option(
            '-s', '--strategy', 'strategies', metavar='STRATEGY',
            type=StrategyParam(),
            multiple=True,
            help="Which optimization strategy will be used. "
                 "Can be 'random', 'ggga', "
                 "or a YAML document describing the strategy."),
        option(
            '--csv', metavar='FILE', type=click.File('w'),
            help="Write evaluation results to a CSV file. "
                 "Only use this when running a single strategy."),
        option(
            '--style', metavar='STYLE',
            help="DualDependenceStyle for the plots.")
    ]

    def decorator(fn):
        for opt in reversed(common_options):
            fn = opt(fn)

        return fn

    return decorator


@click.group()
@click_common_example_options(provide_defaults=True)
@click.pass_context
def cli(ctx, **kwargs):
    """Run an example optimization benchmark function"""

    ctx.obj.update(kwargs)


for example_name, example in EXAMPLES.items():

    if example.variable_dimension:
        assert isinstance(example, ExampleWithVariableDimensions)
        maybe_dim_option = click.option(
            '--dimensions', '-D', type=int,
            default=example.default_dimension,
            show_default=(example.default_dimension is not None),
            help="Number of parameters/dimensions.")
    else:
        def maybe_dim_option(fn):
            return fn

    @cli.command(example_name, help=EXAMPLES[example_name].description)
    @click_common_example_options(provide_defaults=False)
    @maybe_dim_option
    @click.pass_context
    def run_example_wrapper(ctx, **kwargs):
        obj = dict(ctx.obj)
        obj.update((k, v) for k, v in kwargs.items() if v is not None)
        return run_example(ctx.command.name, **obj)

if __name__ == '__main__':
    cli(obj={})  # pylint: disable=no-value-for-parameter; weird click magic

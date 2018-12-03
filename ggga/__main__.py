import abc
import argparse
import asyncio
import json
import typing as t

import attr
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import yaml

from pkg_resources import resource_string

from . import Space, Real, Integer
from . import Minimizer, ObjectiveFunction, RandomState
from . import SurrogateModel, SurrogateModelGPR, SurrogateModelKNN
from .outputs import Output
from .visualization import PartialDependence
from .util import yaml_constructor
from .examples import Example, EXAMPLES

StrategyResult = t.Tuple[
    SurrogateModel, np.ndarray, np.ndarray, float, np.ndarray]


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
        i_best = np.argmin(ys)
        return model, xs, ys, ys[i_best], xs[i_best]


class GGGAStrategy(Strategy):
    name = 'GGGA'

    def __init__(
        self, *,
        minimizer_args: dict = None,
    ) -> None:
        self.minimizer_args = minimizer_args or {}

    @staticmethod
    @yaml_constructor('!GGGA', safe=True)
    def from_yaml(loader: yaml.Loader, node) -> 'GGGAStrategy':
        args = loader.construct_mapping(node)
        return GGGAStrategy(minimizer_args=args)

    async def run(
        self,
        objective: ObjectiveFunction, *,
        cfg: StrategyConfiguration,
        rng: RandomState,
    ) -> StrategyResult:
        minimizer = Minimizer(
            max_nevals=cfg.n_samples,
            surrogate_model_class=cfg.surrogate_model_class,
            **self.minimizer_args,
        )
        res = await minimizer.minimize(
            objective, space=cfg.space, rng=rng,
            outputs=(
                Output(space=cfg.space, log_file=None)
                if cfg.quiet else None),
        )
        best = res.best_individual
        return res.model, res.xs, res.ys, best.observation, best.sample


@attr.s
class IraceStrategy(Strategy):
    name = 'irace'

    port: int = attr.ib()
    parallel: int = 1
    digits: int = 4
    min_racing_rounds: int = 2
    confidence: float = 0.95

    @staticmethod
    @yaml_constructor('!Irace', safe=True)
    def from_yaml(loader: yaml.Loader, node) -> 'IraceStrategy':
        args = loader.construct_mapping(node)
        return IraceStrategy(**args)

    async def _run_evaluation_server(
            self, objective: ObjectiveFunction, *,
            space: Space,
            server_has_started: asyncio.Event,
            irace_has_completed: asyncio.Event,
    ) -> t.Tuple[t.List[list], t.List[float]]:
        all_x = []
        all_y = []

        async def handle_request(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            firstline = (await reader.readline()).decode().rstrip('\r\n')
            if firstline != "evaluation request":
                raise RuntimeError(
                    f"evaluation request mismatch: {firstline!r}")
            # print(f"<<< EVALUATION SERVER >>> received header")

            request = json.loads(await reader.readline())
            # print(f"<<< EVALUATION SERVER >>> received {request}")
            seed = request['seed'][0]
            params = request['params'][0]
            if len(params) != len(space.params):
                raise RuntimeError(
                    f"Parameter mismatch.\n"
                    f"  received: {sorted(params)}\n"
                    f"  expected: {sorted(p.name for p in space.params)}\n")

            x = [params[p.name] for p in space.params]
            y, _cost = await objective(x, RandomState(seed))
            all_x.append(x)
            all_y.append(y)
            # print(f"<<< EVALUATION SERVER >>> f({x}) = {y}")

            writer.write(json.dumps({'y': y}).encode())
            writer.write_eof()

        server = await asyncio.start_server(
            handle_request, host='localhost', port=self.port)
        server_has_started.set()

        await irace_has_completed.wait()

        server.close()
        await server.wait_closed()

        return all_x, all_y

    async def _run_irace_process(
        self, *,
        cfg: StrategyConfiguration,
        seed: int,
        server_has_started: asyncio.Event,
        irace_has_completed: asyncio.Event,
    ) -> None:
        await server_has_started.wait()

        r_code = resource_string('ggga.examples', 'irace_runner.r').decode()

        irace_params = []
        for param in cfg.space.params:
            if isinstance(param, Real):
                irace_params.append(
                    f'{param.name} "" r ({param.lo}, {param.hi})')
            elif isinstance(param, Integer):
                irace_params.append(
                    f'{param.name} "" i ({param.lo}, {param.hi})')
            else:
                raise TypeError(f"Unknown param type: {param}")

        r_code = (r_code
                  .replace('___PORT___', str(self.port))
                  .replace('___N_SAMPLES___', str(cfg.n_samples))
                  .replace('___PARALLEL___', str(self.parallel))
                  .replace('___SEED___', str(seed))
                  .replace('___DIGITS___', str(self.digits))
                  .replace('___FIRST_TEST___', str(self.min_racing_rounds))
                  .replace('___CONFIDENCE___', str(self.confidence))
                  .replace('___PARAMS___', '\n'.join(irace_params)))

        irace = await asyncio.create_subprocess_exec(
            'R', '--no-save',
            stdin=asyncio.subprocess.PIPE, stdout=None, stderr=None)

        irace_stdin = irace.stdin
        assert irace_stdin is not None
        irace_stdin.write(r_code.encode())
        irace_stdin.write_eof()

        await irace.wait()
        irace_has_completed.set()
        if irace.returncode != 0:
            raise RuntimeError(
                f"irace terminated with status {irace.returncode}")

    async def run(
        self,
        objective: ObjectiveFunction, *,
        cfg: StrategyConfiguration,
        rng: RandomState,
    ) -> StrategyResult:

        server_has_started = asyncio.Event()
        irace_has_completed = asyncio.Event()

        (xs, ys), _ = await asyncio.gather(
            self._run_evaluation_server(
                objective, space=cfg.space,
                server_has_started=server_has_started,
                irace_has_completed=irace_has_completed,
            ),
            self._run_irace_process(
                cfg=cfg, seed=rng.randint(999999999),
                server_has_started=server_has_started,
                irace_has_completed=irace_has_completed,
            ),
        )

        # ignore the first two evals that were used for checking:
        xs = xs[2:]
        ys = ys[2:]

        # if no evals available, that means the check failed
        if not ys:
            raise RuntimeError(f"Irace didn't perform any evaluations!")

        model = cfg.surrogate_model_class.estimate(
            xs, ys, space=cfg.space, rng=rng, prior=None)
        i_best = np.argmin(ys)
        return model, xs, ys, ys[i_best], xs[i_best]


async def run_example_with_strategies(  # pylint: disable=too-many-locals
    example_name: str, example: Example, *,
    strategies: t.List[Strategy],
    cfg: StrategyConfiguration,
    rng_seed: int,
    log_y: bool,
    noise_level: float,
) -> t.List[plt.Figure]:
    objective = example.make_objective(log_y=log_y, noise_level=noise_level)

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

    strategy_cfg = StrategyConfiguration(
        space=example.space,
        n_samples=options.samples,
        surrogate_model_class=surrogate_model_class,
        quiet=options.quiet,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example_with_strategies(
        options.example, example,
        strategies=strategies,
        cfg=strategy_cfg,
        rng_seed=options.seed,
        log_y=options.log_y,
        noise_level=options.noise,
    ))

    if options.interactive:
        plt.show()


if __name__ == '__main__':
    main()

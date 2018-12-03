import asyncio
import json
import typing as t

from pkg_resources import resource_string

import attr
import numpy as np  # type: ignore

from .. import Space, Real, Integer, ObjectiveFunction, RandomState
from .strategy import Strategy, StrategyConfiguration, StrategyResult


@attr.s
class IraceStrategy(Strategy):
    name = 'irace'

    port: int = attr.ib()
    parallel: int = 1
    digits: int = 4
    min_racing_rounds: int = 2
    confidence: float = 0.95

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

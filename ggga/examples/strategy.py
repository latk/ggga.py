import abc
import typing as t

import attr
import numpy as np  # type: ignore

from .. import SurrogateModel, ObjectiveFunction, RandomState, Space

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

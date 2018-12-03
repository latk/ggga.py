from .. import Minimizer, RandomState, ObjectiveFunction, Output
from .strategy import Strategy, StrategyConfiguration, StrategyResult


class GGGAStrategy(Strategy):
    name = 'GGGA'

    def __init__(
        self, *,
        minimizer_args: dict = None,
    ) -> None:
        self.minimizer_args = minimizer_args or {}

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

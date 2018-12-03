import typing as t

import attr
import numpy as np  # type: ignore

from .. import ObjectiveFunction, Space, Real
from ..space import Log1pScale
from ..benchmark_functions import (
    goldstein_price, easom, himmelblau, rastrigin, rosenbrock, sphere, onemax,
    trap)


@attr.s
class Example:
    function: t.Callable[..., float] = attr.ib()
    space: Space = attr.ib()
    minima: t.List[t.Tuple[list, float]] = attr.ib()

    def make_objective(
        self, *, log_y: bool, noise_level: float,
    ) -> ObjectiveFunction:

        async def objective(xs, rng):
            y = self.function(*xs)

            if noise_level != 0.0:
                noise = noise_level * rng.standard_normal()
                while y + noise < 0:
                    noise = noise_level * rng.standard_normal()
                y += noise

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

EXAMPLES['onemax4'] = Example(
    function=onemax,
    space=Space(
        Real('x_1', 0.0, 1.0),
        Real('x_2', 0.0, 1.0),
        Real('x_3', 0.0, 1.0),
        Real('x_4', 0.0, 1.0),
    ),
    minima=[([0.0]*4, 0.0)],
)

EXAMPLES['onemax4log'] = Example(
    function=onemax,
    space=Space(
        Real('x_1', 0.0, 1.0, scale=Log1pScale(2)),
        Real('x_2', 0.0, 1.0, scale=Log1pScale(2)),
        Real('x_3', 0.0, 1.0, scale=Log1pScale(2)),
        Real('x_4', 0.0, 1.0, scale=Log1pScale(2)),
    ),
    minima=[([0.0]*4, 0.0)],
)

EXAMPLES['trap2'] = Example(
    function=trap,
    space=Space(
        Real('x_1', -1.0, 1.0),
        Real('x_2', -1.0, 1.0),
    ),
    minima=[([0.0]*2, 0.0)],
)


EXAMPLES['trap4'] = Example(
    function=trap,
    space=Space(
        Real('x_1', -1.0, 1.0),
        Real('x_2', -1.0, 1.0),
        Real('x_3', -1.0, 1.0),
        Real('x_4', -1.0, 1.0),
    ),
    minima=[([0.0]*4, 0.0)],
)

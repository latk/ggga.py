import inspect
import typing as t

import attr
import numpy as np  # type: ignore

from .. import ObjectiveFunction, Space, Real
from ..space import Log1pScale
from ..benchmark_functions import (
    goldstein_price, easom, himmelblau, rastrigin, rosenbrock, sphere, onemax,
    trap)


def _get_doc(fn: t.Callable) -> str:
    rawdoc = inspect.getdoc(fn) or ""
    lines = rawdoc.splitlines() or [""]
    return lines[0]


@attr.s
class Example:
    variable_dimension: bool = False
    function: t.Callable[..., float] = attr.ib()
    space: Space = attr.ib()
    minima: t.List[t.Tuple[list, float]] = attr.ib()
    description: str = attr.ib(
        default=attr.Factory(
            lambda self: _get_doc(self.function),
            takes_self=True)
    )

    def fix_dimension(self, n_dim: t.Optional[int]) -> 'Example':
        if n_dim is not None:
            raise TypeError(
                "dimension of Example cannot be changed afterwards")

        return self

    def make_objective(
        self, *,
        log_y: bool,
        noise_level: float,
        on_evaluation: t.Callable[[list, float], None] = None,
    ) -> ObjectiveFunction:

        async def objective(xs, rng):
            y = self.function(*xs)

            if noise_level != 0.0:
                noise = noise_level * rng.standard_normal()
                while y + noise < 0:
                    noise = noise_level * rng.standard_normal()
                y += noise

            if on_evaluation is not None:
                on_evaluation(xs, y)

            if log_y:
                assert y > 0, f"ys must be positive, was {y}"
                y = np.log(y)

            cost = 0.0
            return y, cost

        return objective


@attr.s
class ExampleWithVariableDimensions:
    variable_dimension: bool = True
    function: t.Callable[..., float] = attr.ib()
    make_space: t.Callable[[int], Space] = attr.ib()
    make_mimima: t.Callable[[int], t.List[t.Tuple[list, float]]] = attr.ib()
    default_dimension: t.Optional[int] = attr.ib()
    description: str = attr.ib(
        default=attr.Factory(
            lambda self: _get_doc(self.function),
            takes_self=True))

    def fix_dimension(self, n_dim: t.Optional[int]) -> Example:
        if n_dim is None or n_dim < 1:
            raise ValueError(
                f"at least one dimension is required, "
                f"but got n_dim = {n_dim}")

        return Example(
            function=self.function,
            space=self.make_space(n_dim),
            minima=self.make_mimima(n_dim),
            description=self.description,
        )


EXAMPLES: t.Dict[str, t.Union[Example, ExampleWithVariableDimensions]] = {}

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
    description=_get_doc(easom),
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

EXAMPLES['rastrigin'] = ExampleWithVariableDimensions(
    function=rastrigin,
    default_dimension=2,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', -5.12, 5.12) for x in range(1, n_dim + 1)],
    ),
    make_mimima=lambda n_dim: [([0.0]*n_dim, 0.0)],
)

EXAMPLES['rosenbrock'] = ExampleWithVariableDimensions(
    function=rosenbrock,
    default_dimension=2,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', -5.12, 5.12) for x in range(1, n_dim + 1)],
    ),
    make_mimima=lambda n_dim: [([1.0]*n_dim, 0.0)],
)

EXAMPLES['sphere'] = ExampleWithVariableDimensions(
    function=sphere,
    default_dimension=2,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', -2.0, 2.0) for x in range(1, n_dim + 1)]
    ),
    make_mimima=lambda n_dim: [([0.0]*n_dim, 0.0)],
)

EXAMPLES['onemax'] = ExampleWithVariableDimensions(
    function=onemax,
    default_dimension=4,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', 0.0, 1.0) for x in range(1, n_dim + 1)],
    ),
    make_mimima=lambda n_dim: [([0.0]*n_dim, 0.0)],
)

EXAMPLES['onemax-log'] = ExampleWithVariableDimensions(
    function=onemax,
    default_dimension=4,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', 0.0, 1.0, scale=Log1pScale(2))
            for x in range(1, n_dim + 1)],
    ),
    make_mimima=lambda n_dim: [([0.0]*n_dim, 0.0)],
)

EXAMPLES['trap'] = ExampleWithVariableDimensions(
    function=trap,
    default_dimension=2,
    make_space=lambda n_dim: Space(
        *[Real(f'x_{x}', -1.0, 1.0) for x in range(1, n_dim + 1)],
    ),
    make_mimima=lambda n_dim: [([0.0]*n_dim, 0.0)],
)

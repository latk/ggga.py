#!/usr/bin/env python3

import argparse
import asyncio

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from ggga.benchmark_functions import goldstein_price
from ggga import Space, Real, SurrogateModelGPR, minimize
import ggga.visualization as viz


SPACE = Space(
    Real('x_1', '--x1', -2, 2),
    Real('x_2', '--x2', -2, 2),
)
X_MIN = (0.0, -1.0)


async def run_example(*, rng: RandomState, n_samples: int, log_y: bool):

    async def objective(xs, rng):
        y = goldstein_price(xs[0], xs[1])

        if log_y:
            assert y > 0, f"ys must be positive, was {y}"
            y = np.log(y)

        cost = 0.0
        return y, cost

    # evaluate a couple of random samples
    xs = np.array([SPACE.sample(rng=rng) for _ in range(n_samples)])
    ys = goldstein_price(xs[:, 0], xs[:, 1])
    if log_y:
        ys = np.log(ys)
    model = SurrogateModelGPR.estimate(
        xs, ys, space=SPACE, rng=rng, prior=None)
    y_min, y_min_std = model.predict(X_MIN)

    print(f"Minimum after {n_samples} random samples: (None), "
          f"f({X_MIN}) = {y_min:.2f} +/- {y_min_std:.2f}")

    fig, _ = viz \
        .PartialDependence(model=model, space=SPACE, rng=rng) \
        .plot_grid(xs, ys)
    fig.suptitle(f"Goldstein-Price ({n_samples} random samples)")

    # do a proper GGGA-run
    res = await minimize(objective, space=SPACE, max_nevals=n_samples, rng=rng)
    y_min, y_min_std = res.model.predict(X_MIN)

    print(f"Minimum after {n_samples} GGGA-samples: {res.fmin:.2f}, "
          f"f({X_MIN}) = {y_min:.2f} +/- {y_min_std:.2f}")

    fig, _ = viz \
        .PartialDependence(model=res.model, space=SPACE, rng=rng) \
        .plot_grid(res.xs, res.ys)
    fig.suptitle(f"Goldstein-Price ({n_samples} GGGA-samples)")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.description = r"""
    Example running on the Goldstein-Price benchmark function.
    The optimium is at f(0, -1) = 3.
    """

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

    return parser


def main() -> None:
    options = make_argument_parser().parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_example(
        rng=RandomState(7861),
        n_samples=options.samples,
        log_y=options.log_y,
    ))
    if options.interactive:
        plt.show()


if __name__ == '__main__':
    main()
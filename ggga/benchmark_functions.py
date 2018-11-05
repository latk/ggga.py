import numpy as np  # type: ignore


def goldstein_price(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
    r"""Goldstein-Price: Asymetric function with single optimum.

    Bounds: -2 <= x_1, x_2 <= 2

    Optimium: f(0,-1) = 3

    >>> goldstein_price(0.0, -1.0)
    3.0

    Definition taken from:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    return (1
            * (1 + (x_1 + x_2 + 1)**2
                * (19 - 14*x_1 + 3*x_1**2 - 14*x_2 + 6*x_1*x_2 + 3*x_2**2))
            * (30 + (2*x_1 - 3*x_2)**2
                * (18 - 32*x_1 + 12*x_1**2 + 48*x_2 - 36*x_1*x_2 + 27*x_2**2)))


def easom(
    x_1: np.ndarray, x_2: np.ndarray, *,
    amplitude: float = 1.0,
) -> np.ndarray:
    r"""Easom: Flat function with single sharp minimum.

    Bounds: -50 <= x_1, x_2 <= 50 or other.

    Optimum: f(pi,pi) = 0

    >>> easom(np.pi, np.pi)
    0.0
    >>> round(easom(0, 0), 5)
    1.0

    Definition taken from:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    and adapted for non-negative outputs.
    """

    return amplitude * (1 - np.cos(x_1) * np.cos(x_2) * np.exp(
        -((x_1 - np.pi)**2 + (x_2 - np.pi)**2)
    ))


def himmelblau(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
    r"""Himmelblau's function: Asymetric polynomial with 4 minima.

    Bounds: -5 <= x_1, x_2 <= 5

    Minima:

    >>> round(himmelblau(3.0, 2.0), 5)
    0.0
    >>> round(himmelblau(-2.805118, 3.131312), 5)
    0.0
    >>> round(himmelblau(-3.779310, -3.283186), 5)
    0.0
    >>> round(himmelblau(3.584428, -1.848126), 5)
    0.0
    """

    return (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2


def rastrigin(*xs: np.ndarray, amplitude: float = 10) -> np.ndarray:
    r"""Rastrigin Function: N-dimensional with many local minima.

    Bounds: -5.12 <= xi <= 5.12

    Optimum: f(0, ..., 0) = 0

    >>> rastrigin(0.0)  # minimum in 1D
    0.0
    >>> rastrigin(0.0, 0.0)  # minimum in 2D
    0.0
    >>> rastrigin(*[0.0]*10)  # minimum in 10D
    0.0

    >>> rastrigin()
    Traceback (most recent call last):
    TypeError: ...
    """

    if not xs:
        raise TypeError("at least one dimension required")

    assert amplitude > 0.0

    n_dim = len(xs)

    return amplitude * n_dim + sum(
        x_i**2 - amplitude * np.cos(2 * np.pi * x_i)
        for x_i in xs
    )


def rosenbrock(*xs: np.ndarray) -> np.ndarray:
    r"""Rosenbrock function: N-dimensional and asymetric.

    Bounds: unbounded, but interval [-2.5, 2.5] sensible

    Optimum: f(1, ..., 1) = 0

    >>> rosenbrock(1.0, 1.0)
    0.0
    >>> rosenbrock(*[1.0]*6)
    0.0
    """

    if len(xs) < 2:
        raise TypeError("at least two dimensions required")

    return sum(
        100 * (xs[i + 1] - xs[i]**2)**2 + (1 - xs[i])**2
        for i in range(len(xs) - 1)
    )


def sphere(*xs: np.ndarray) -> np.ndarray:
    r"""Sphere function: N-dimensional, symmetric.

    Bounds: unbounded, but -2 <= xi <= 2 is sensible.

    Optimum: f(0, ..., 0) = 0

    >>> sphere(*[0.0]*1)
    0.0
    >>> sphere(*[0.0]*2)
    0.0
    >>> sphere(*[0.0]*6)
    0.0
    """

    if not xs:
        raise TypeError("at least one dimension required")

    return sum(x**2 for x in xs)


def onemax(*xs: np.ndarray) -> float:
    r"""One-Max function.

    Bounds: 0 <= xi <= 1

    Optimum: f(0, ..., 0) = 0

    >>> onemax(*[0.0]*1)
    0.0
    >>> onemax(*[0.0]*2)
    0.0
    >>> onemax(*[0.0]*6)
    0.0

    >>> onemax(*[1.0]*1)
    1.0
    >>> onemax(*[1.0]*6)
    6.0
    """

    if not xs:
        raise TypeError("at least one dimension required")

    return sum(x for x in xs)

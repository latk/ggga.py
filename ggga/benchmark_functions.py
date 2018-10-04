import numpy as np  # type: ignore


def goldstein_price(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
    r"""
    Asymetric function with single optimum.

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

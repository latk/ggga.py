import abc
import typing as t
import warnings

from numpy.random import RandomState  # type: ignore
import numpy as np  # type: ignore
import scipy.optimize  # type: ignore

TNumpy = t.TypeVar('TNumpy', np.ndarray, float)


def fork_random_state(rng):
    return RandomState(rng.randint(2**32 - 1))


def tabularize(
        header: t.List[str],
        formats: t.List[str],
        data: t.Iterable[list],
) -> str:
    assert len(header) == len(formats), (header, formats)

    columns = [[str(h)] for h in header]
    for row in data:
        for col, fmt, data_item in zip(columns, formats, row):
            col.append(fmt.format(data_item))
    assert all(len(columns[0]) == len(col) for col in columns), \
        [len(col) for col in columns]
    col_size = [max(len(d) for d in col) for col in columns]
    out = []
    out.append(' '.join('-' * size for size in col_size))
    for i in range(len(columns[0])):
        out.append(' '.join(
            col[i].rjust(size) for col, size in zip(columns, col_size)))
    out[0], out[1] = out[1], out[0]
    return '\n'.join(out)


def minimize_by_gradient(
    obj_func: t.Callable,
    start: np.ndarray,
    *,
    bounds: list = None,
    approx_grad: bool = False,
) -> t.Tuple[np.ndarray, float]:
    result, fmin, convergence_dict = scipy.optimize.fmin_l_bfgs_b(
        obj_func, start, bounds=bounds, approx_grad=approx_grad)

    if convergence_dict['warnflag'] != 0:
        warnings.warn(
            f"fmin_l_bfgs_b failed with state:\n"
            f"        {convergence_dict}")

    # clip values to bounds, if any
    if bounds is not None:
        lo, hi = np.array(bounds).T
        result = np.clip(result, lo, hi)

    return result, fmin


def timer(time_source) -> t.Callable[[], float]:
    start = time_source()

    def duration():
        return time_source() - start

    return duration


class ToJsonish(abc.ABC):
    @abc.abstractmethod
    def to_jsonish(self) -> object:
        pass


def coerce_array(
    arrayish: t.Union[np.ndarray, t.Iterable],
) -> np.ndarray:
    if isinstance(arrayish, np.ndarray):
        return arrayish
    return np.array(arrayish)

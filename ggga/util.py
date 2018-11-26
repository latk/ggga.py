import abc
import typing as t
import warnings

from numpy.random import RandomState  # type: ignore
import numpy as np  # type: ignore
import scipy.optimize  # type: ignore
import yaml

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
    result, fmin, info = scipy.optimize.fmin_l_bfgs_b(
        obj_func, start, bounds=bounds, approx_grad=approx_grad)

    line_search_task = b'ABNORMAL_TERMINATION_IN_LNSRCH'
    if info['warnflag'] == 2 and info['task'] == line_search_task:
        # This indicates an inconsistent gradient. This is very unfortunate,
        # but it is not clear why the gradient should be wrong.
        # For background see https://stackoverflow.com/a/39155976/1521179
        pass

    elif info['warnflag'] != 0:
        warnings.warn(
            f"fmin_l_bfgs_b failed with state:\n"
            f"        {info}")

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


def yaml_constructor(tag, safe=False):
    def decorator(constructor):
        yaml.add_constructor(
            tag, constructor,
            Loader=(yaml.SafeLoader if safe else yaml.Loader),
        )
        return constructor

    return decorator


class Validator:

    @staticmethod
    def is_instance(expected_type=None):
        def validate(_obj, attr, value):
            if expected_type is None:
                my_expected_type = attr.type
            else:
                my_expected_type = expected_type
            if not isinstance(value, my_expected_type):
                typename = getattr(my_expected_type, '__qualname__')
                raise TypeError(f"{attr.name} must be instance of {typename}")

        return validate

    @staticmethod
    def is_posint(_obj, attr, value):
        if not (isinstance(value, int) and value > 0):
            raise TypeError(f"{attr.name} must be a positive integer")

    @staticmethod
    def is_percentage(_obj, attr, value):
        if not (isinstance(value, float) and 0.0 <= value <= 1.0):
            raise TypeError(f"{attr.name} must be a fraction from 0.0 to 1.0")

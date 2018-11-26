import abc
import typing as t

import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

T = t.TypeVar('T')


class Param(abc.ABC, t.Generic[T]):
    def __init__(
        self, name: str, *, fmt: str,
    ) -> None:
        self.name: str = name
        self.fmt: str = fmt

    @abc.abstractmethod
    def sample(
        self, *, rng: RandomState, lo: T = None, hi: T = None,
    ) -> T:
        raise NotImplementedError

    @abc.abstractproperty
    def size(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self, value: T) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid_transformed(self, value: float) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def into_transformed(self, value: T) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def from_transformed(self, value: float) -> T:
        raise NotImplementedError

    def into_transformed_a(self, values: t.List[T]) -> t.List[float]:
        return [self.into_transformed(x) for x in values]

    def from_transformed_a(self, values: t.List[float]) -> t.List[T]:
        return [self.from_transformed(x) for x in values]

    @abc.abstractmethod
    def transformed_bounds(self) -> t.Tuple[float, float]:
        raise NotImplementedError

    def bounds(self) -> t.Tuple[T, T]:
        raise NotImplementedError


class Integer(Param[int]):
    lo: int
    hi: int

    def __init__(
        self, name: str, lo: int, hi: int, *,
        fmt: str = '{}',
    ) -> None:
        super().__init__(name, fmt=fmt)
        self.lo = lo
        self.hi = hi

    def __repr__(self) -> str:
        return f'Integer({self.name!r}, {self.lo!r}, {self.hi!r})'

    def sample(
        self, *,
        rng: RandomState,
        lo: int = None,
        hi: int = None,
    ) -> int:
        if lo is None:
            lo = self.lo
        else:
            assert self.lo <= lo
        if hi is None:
            hi = self.hi
        else:
            assert hi <= self.hi
        return rng.randint(lo, hi + 1)

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1

    def is_valid(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    @staticmethod
    def is_valid_transformed(value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: int) -> float:
        return (value - self.lo + 0.5) / self.size

    def from_transformed(self, value: float) -> int:
        return int(np.round(value * self.size - 0.5)) + self.lo

    @staticmethod
    def transformed_bounds() -> t.Tuple[float, float]:
        return (0.0, 1.0)

    def bounds(self) -> t.Tuple[int, int]:
        return self.lo, self.hi


class Scale(abc.ABC):
    @abc.abstractmethod
    def transform(self, x: float) -> float:
        pass

    @abc.abstractmethod
    def reverse(self, x: float) -> float:
        pass


class Real(Param[float]):
    lo: float
    hi: float

    def __init__(
        self, name: str, lo: float, hi: float, *,
        scale: Scale = None,
        fmt: str = '{:.5f}',
    ) -> None:
        super().__init__(name, fmt=fmt)
        self.lo: float = lo
        self.hi: float = hi
        self.scale: t.Optional[Scale] = scale

        if scale is not None:
            assert isinstance(scale, Scale)

    def __repr__(self) -> str:
        lo, hi = self.lo, self.hi
        return f'Integer({self.name!r}, {lo!r}, {hi!r}), scale={self.scale!r})'

    def sample(
        self, *,
        rng: RandomState,
        lo: float = None,
        hi: float = None,
    ) -> float:
        if lo is None:
            lo = self.lo
        else:
            assert self.lo <= lo
        if hi is None:
            hi = self.hi
        else:
            assert hi <= self.hi
        return self.from_transformed(self.sample_transformed(
            rng=rng,
            lo=self.into_transformed(lo),
            hi=self.into_transformed(hi),
        ))

    @staticmethod
    def sample_transformed(
        *, rng: RandomState, lo: float, hi: float,
    ) -> float:
        assert 0.0 <= lo <= hi <= 1.0, \
            f'bounds [{lo},{hi}] must be within [0,1]'
        size = hi - lo
        return rng.random_sample() * size + lo

    @property
    def size(self) -> float:
        return self.hi - self.lo

    def is_valid(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    @staticmethod
    def is_valid_transformed(value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: float) -> float:
        x = value

        x = (x - self.lo) / self.size

        if self.scale is not None:
            x = self.scale.transform(x)

        return x

    def from_transformed(self, value: float) -> float:
        x = value

        if self.scale is not None:
            x = self.scale.reverse(x)

        x = x * self.size + self.lo

        return x

    @staticmethod
    def transformed_bounds() -> t.Tuple[float, float]:
        return (0.0, 1.0)

    def bounds(self) -> t.Tuple[float, float]:
        return self.lo, self.hi


class Log1pScale(Scale):
    def __init__(self, scale: float) -> None:
        self._scale = np.exp(scale)
        self._output_size = np.log1p(self._scale * 1)

    def transform(self, x: float) -> float:
        assert 0.0 <= x <= 1.0, f'value {x} should be in [0, 1]'
        return np.log1p(self._scale * x) / self._output_size

    def reverse(self, x: float) -> float:
        assert 0.0 <= x <= 1.0, f'value {x} should be in [0, 1]'
        return np.expm1(x * self._output_size) / self._scale


Sample = t.List[float]
ConstraintFunction = t.Callable[[Sample], bool]
BoundSuggestionFunction = t.Callable[[Sample], dict]


class Space:

    def __init__(
        self, *params: Param,
        constraints: t.List[ConstraintFunction] = None,
        constrained_bounds_suggestions: t.List[BoundSuggestionFunction] = None,
    ) -> None:
        if constraints is None:
            constraints = []
        if constrained_bounds_suggestions is None:
            constrained_bounds_suggestions = []

        self.params = params
        self.constraints = constraints
        self.constrained_bounds_suggestions = constrained_bounds_suggestions

        seen_names: t.Set[str] = set()
        for param in params:
            assert isinstance(param, Param), f'must be a param: {param!r}'
            assert param.name not in seen_names, \
                f'param names must be unique: {param.name}'
            seen_names.add(param.name)

        assert all(callable(c) for c in constraints)
        assert all(callable(s) for s in constrained_bounds_suggestions)

    def __repr__(self):
        out = 'Space('
        for param in self.params:
            out += f'\n  {param!r},'
        out += '\n  constraints=['
        for constraint in self.constraints:
            out += f'\n    {constraint}'
        out += '],'
        out += '\n  constrained_bounds_suggestions=['
        for suggestion in self.constrained_bounds_suggestions:
            out += f'\n    {suggestion}'
        out += '],'
        out += '\n)'
        return out

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self, *, rng: RandomState) -> Sample:
        retries = 10
        bounds: t.Dict[str, tuple] = dict()

        def merge_intervals(*intervals):
            lows, highs = zip(intervals)
            lo = max((lo for lo in lows if lo is not None), default=None)
            hi = max((hi for hi in highs if hi is not None), default=None)

            if lo is not None and hi is not None:
                assert lo <= hi

            return lo, hi

        for _ in range(retries):
            the_sample = []

            for param in self.params:
                lo, hi = bounds.get(param.name, (None, None))
                the_sample.append(param.sample(rng=rng, lo=lo, hi=hi))

            if all(c(the_sample) for c in self.constraints):
                return the_sample

            for suggestion in self.constrained_bounds_suggestions:
                for name, suggested_bounds in suggestion(the_sample).items():
                    if suggested_bounds is None:
                        continue
                    old_bounds = bounds.get(name, (None, None))
                    bounds[name] = merge_intervals(
                        old_bounds, suggested_bounds)

        raise RuntimeError("Could not find valid sample")

    def mutate(
        self, sample: list, *,
        rng: RandomState,
        relscale: t.Union[float, t.Iterable[float]],
    ) -> list:
        return self.from_transformed(
            self.mutate_transformed(
                self.into_transformed(sample),
                rng=rng, relscale=relscale))

    def mutate_transformed(
        self, sample_transformed: list, *,
        rng: RandomState,
        relscale: t.Union[float, t.Iterable[float]],
    ) -> list:
        if not isinstance(relscale, t.Iterable):
            relscale = [relscale] * self.n_dims
        cov = np.diag(relscale)
        retries = int(20 * np.sqrt(self.n_dims))
        for _ in range(retries):
            mut_sample = rng.multivariate_normal(sample_transformed, cov)
            if self.is_valid_transformed(mut_sample):
                return mut_sample
            cov *= 0.9  # make feasibility more likely
        raise RuntimeError(
            f"mutation failed to produce values within bounds"
            f"\n  last mut_sample = {mut_sample}"
            f"\n  input sample    = {sample_transformed}")

    def is_valid(self, sample) -> bool:
        return all(p.is_valid(v) for p, v in zip(self.params, sample)) \
                and all(c(sample) for c in self.constraints)

    def is_valid_transformed(self, sample: list) -> bool:
        return all(p.is_valid_transformed(v)
                   for p, v in zip(self.params, sample))

    def into_transformed(self, sample: list) -> list:
        return [p.into_transformed(v) for p, v in zip(self.params, sample)]

    def from_transformed(self, sample: list) -> list:
        return [p.from_transformed(v) for p, v in zip(self.params, sample)]

    @property
    def param_names(self) -> t.List[str]:
        return [p.name for p in self.params]

import abc
import typing as t
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

T = t.TypeVar('T')


class Param(abc.ABC, t.Generic[T]):
    def __init__(self, name, flag):
        self.name = name
        self.flag = flag

    @abc.abstractmethod
    def sample(self, *, rng: RandomState, lo=None, hi=None) -> T:
        pass

    @abc.abstractmethod
    def mutate(self, value: T, *, rng: RandomState, relscale: float) -> T:
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractmethod
    def is_valid(self, value: T) -> bool:
        pass

    @abc.abstractmethod
    def is_valid_transformed(self, value) -> bool:
        pass

    @abc.abstractmethod
    def into_transformed(self, value: T):
        pass

    @abc.abstractmethod
    def from_transformed(self, value) -> T:
        pass

    def into_transformed_a(self, values: list) -> list:
        return [self.into_transformed(x) for x in values]

    def from_transformed_a(self, values: list) -> list:
        return [self.from_transformed(x) for x in values]

    @abc.abstractmethod
    def transformed_bounds(self) -> tuple:
        pass

    def bounds(self) -> t.Optional[t.Tuple[T, T]]:
        return None


class Integer(Param[int]):
    lo: int
    hi: int

    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(
        self, *,
        rng: RandomState,
        lo: int=None,
        hi: int=None,
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

    def mutate(self, value, *, rng: RandomState, relscale: float) -> int:
        retries = 20
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
            relscale *= 0.8
        raise RuntimeError("mutation failed to produce values within bounds")

    @property
    def size(self) -> int:
        return self.hi - self.lo

    def is_valid(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def is_valid_transformed(self, value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: int) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> int:
        return int(np.round(value * self.size + self.lo))

    def transformed_bounds(self) -> t.Tuple[float, float]:
        return (0.0, 1.0)

    def bounds(self) -> t.Tuple[int, int]:
        return self.lo, self.hi


class Real(Param[float]):
    lo: float
    hi: float

    def __init__(self, name, flag, lo, hi):
        super().__init__(name, flag)
        self.lo = lo
        self.hi = hi

    def sample(
        self, *,
        rng: RandomState,
        lo: float=None,
        hi: float=None,
    ) -> float:
        if lo is None:
            lo = self.lo
        else:
            assert self.lo <= lo
        if hi is None:
            hi = self.hi
        else:
            assert hi <= self.hi
        size = hi - lo
        return rng.random_sample() * size + lo

    def mutate(self, value, *, rng: RandomState, relscale: float) -> float:
        retries = 20
        x = self.into_transformed(value)
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if self.is_valid_transformed(mutx):
                return self.from_transformed(mutx)
            relscale *= 0.8
        raise RuntimeError("mutation failed to produce values within bounds")

    @property
    def size(self) -> float:
        return self.hi - self.lo

    def is_valid(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def is_valid_transformed(self, value: float) -> bool:
        return 0.0 <= value <= 1.0

    def into_transformed(self, value: float) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> float:
        return value * self.size + self.lo

    def transformed_bounds(self) -> t.Tuple[float, float]:
        return (0.0, 1.0)

    def bounds(self) -> t.Tuple[float, float]:
        return self.lo, self.hi


class Space(object):
    def __init__(
        self, *params: Param,
        constraints: t.List[t.Callable[[list], bool]]=None,
        constrained_bounds_suggestions: t.List[t.Callable[[list], dict]]=None,
    ) -> None:
        if constraints is None:
            constraints = []
        if constrained_bounds_suggestions is None:
            constrained_bounds_suggestions = []

        assert all(isinstance(p, Param) for p in params)
        assert all(callable(c) for c in constraints)
        assert all(callable(s) for s in constrained_bounds_suggestions)

        self.params = params
        self.constraints = constraints
        self.constrained_bounds_suggestions = constrained_bounds_suggestions

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self, *, rng: RandomState) -> list:
        retries = 10
        bounds: t.Dict[str, tuple] = dict()

        def merge_lo_hi(llo, lhi, rlo, rhi):
            if   llo is None:   lo = rlo            # noqa
            elif rlo is None:   lo = llo            # noqa
            else:               lo = max(llo, rlo)  # noqa

            if   lhi is None:   hi = rhi            # noqa
            elif rhi is None:   hi = lhi            # noqa
            else:               hi = min(lhi, rhi)  # noqa

            if lo is not None and hi is not None:
                assert lo <= hi
            return lo, hi

        for _ in range(retries):
            s = []
            for param in self.params:
                lo, hi = bounds.get(param.name, (None, None))
                s.append(param.sample(rng=rng, lo=lo, hi=hi))
            if all(c(s) for c in self.constraints):
                return s
            for suggestion in self.constrained_bounds_suggestions:
                for k, v in suggestion(s).items():
                    if v is None:
                        continue
                    llo, lhi = bounds.get(k, (None, None))
                    rlo, rhi = v
                    bounds[k] = merge_lo_hi(llo, lhi, rlo, rhi)

        raise RuntimeError("Could not find valid sample")

    def mutate(self, sample: list, *,
               rng: RandomState, relscale: float) -> list:
        return [
            p.mutate(x, rng=rng, relscale=relscale)
            for p, x in zip(self.params, sample)
        ]

    def is_valid(self, sample) -> bool:
        return all(p.is_valid(v) for p, v in zip(self.params, sample)) \
                and all(c(sample) for c in self.constraints)

    def into_transformed(self, sample: list) -> list:
        return [p.into_transformed(v) for p, v in zip(self.params, sample)]

    def from_transformed(self, sample: list) -> list:
        return [p.from_transformed(v) for p, v in zip(self.params, sample)]

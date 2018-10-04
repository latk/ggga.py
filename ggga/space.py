import abc
import typing as t
import numpy as np  # type: ignore
from numpy.random import RandomState  # type: ignore

T = t.TypeVar('T')
ProjectedFloat = t.NewType('ProjectedFloat', float)


class ProjectedParam:

    def sample(
        self, *,
        rng: RandomState,
        lo: ProjectedFloat = ProjectedFloat(0.0),
        hi: ProjectedFloat = ProjectedFloat(1.0),
    ) -> ProjectedFloat:
        assert 0.0 <= lo <= hi <= 1.0, \
            f"bounds [{lo},{hi}] must be within [0,1]"

        size = hi - lo
        return rng.random_sample() * size + lo

    def mutate(
        self, x: ProjectedFloat, *,
        rng: RandomState,
        relscale: float,
        relscale_attenuation: float = 0.8,
        retries: int = 20,
    ) -> ProjectedFloat:
        for _ in range(retries):
            mutx = x + rng.standard_normal() * relscale
            if 0.0 <= mutx <= 1.0:
                return mutx
            relscale *= relscale_attenuation

        raise RuntimeError(
            f"mutation failed to produce values within bounds"
            f"\n  last mutx = {mutx}"
            f"\n  input x   = {x}")


class Param(abc.ABC, t.Generic[T]):
    def __init__(self, name: str, *, flag: str, fmt: str) -> None:
        self.name: str = name
        self.flag: str = flag
        self.fmt: str = fmt

    @abc.abstractmethod
    def sample(self, *, rng: RandomState, lo=None, hi=None) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_transformed(self, value, *, rng: RandomState, relscale: float):
        raise NotImplementedError

    def mutate(self, value: T, *, rng: RandomState, relscale: float) -> T:
        return self.from_transformed(self.mutate_transformed(
            self.into_transformed(value),
            rng=rng,
            relscale=relscale))

    @abc.abstractproperty
    def size(self):
        raise NotImplementedError

    def is_valid(self, value: T) -> bool:
        return self.is_valid_transformed(self.into_transformed(value))

    def is_valid_transformed(self, value: float) -> bool:
        # pylint: disable=no-self-use
        return 0.0 <= value <= 1.0

    @abc.abstractmethod
    def into_transformed(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def from_transformed(self, value) -> T:
        raise NotImplementedError

    def into_transformed_a(self, values: list) -> list:
        return [self.into_transformed(x) for x in values]

    def from_transformed_a(self, values: list) -> list:
        return [self.from_transformed(x) for x in values]

    def transformed_bounds(self) -> t.Tuple[float, float]:
        # pylint: disable=no-self-use
        return (0.0, 1.0)

    def bounds(self) -> t.Optional[t.Tuple[T, T]]:
        # pylint: disable=no-self-use
        return None


class Integer(Param[int]):
    lo: int
    hi: int

    def __init__(
        self, name: str, flag: str, lo: int, hi: int, *,
        fmt: str = '{}',
    ) -> None:
        super().__init__(name, flag=flag, fmt=fmt)
        self.lo: int = lo
        self.hi: int = hi

    def __repr__(self) -> str:
        return (f'Integer('
                f'{self.name!r}, {self.flag!r}, '
                f'{self.lo!r}, {self.hi!r})')

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

    def mutate_transformed(
        self, x: float, *, rng: RandomState, relscale: float
    ) -> float:
        # pylint: disable=no-self-use
        return ProjectedParam().mutate(
            ProjectedFloat(x), rng=rng, relscale=relscale)

    @property
    def size(self) -> int:
        return self.hi - self.lo

    def is_valid(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def into_transformed(self, value: int) -> float:
        return (value - self.lo) / self.size

    def from_transformed(self, value: float) -> int:
        return int(np.round(value * self.size + self.lo))

    def bounds(self) -> t.Tuple[int, int]:
        return self.lo, self.hi


class Scale(abc.ABC):
    @abc.abstractmethod
    def transform(self, x: ProjectedFloat) -> ProjectedFloat:
        raise NotImplementedError

    @abc.abstractmethod
    def reverse(self, x: ProjectedFloat) -> ProjectedFloat:
        raise NotImplementedError


class Real(Param[float]):
    lo: float
    hi: float

    def __init__(
        self, name: str, flag: str, lo: float, hi: float, *,
        scale: Scale = None,
        fmt: str = '{:.5f}',
    ) -> None:
        super().__init__(name, flag=flag, fmt=fmt)
        self.lo: float = lo
        self.hi: float = hi
        self.scale: t.Optional[Scale] = scale

        if scale is not None:
            assert isinstance(scale, Scale)

    def __repr__(self) -> str:
        return (f'Integer('
                f'{self.name!r}, {self.flag!r}, '
                f'{self.lo!r}, {self.hi!r}), '
                f'scale={self.scale!r}')

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
        return self.from_transformed(ProjectedParam().sample(
            rng=rng,
            lo=self.into_transformed(lo),
            hi=self.into_transformed(hi),
        ))

    def mutate_transformed(
        self, x: float, *, rng: RandomState, relscale: float
    ) -> float:
        # pylint: disable=no-self-use
        return ProjectedParam().mutate(
            ProjectedFloat(x), rng=rng, relscale=relscale)

    @property
    def size(self) -> float:
        return self.hi - self.lo

    def into_transformed(self, value: float) -> ProjectedFloat:
        x = ProjectedFloat((value - self.lo) / self.size)

        if self.scale is not None:
            x = self.scale.transform(x)

        return x

    def from_transformed(self, value: ProjectedFloat) -> float:
        x = value

        if self.scale is not None:
            x = self.scale.reverse(x)

        return x * self.size + self.lo

    def bounds(self) -> t.Tuple[float, float]:
        return self.lo, self.hi


class Log1pScale(Scale):
    def __init__(self, scale: float) -> None:
        self._scale = np.exp(scale)
        self._output_size = np.log1p(self._scale * 1)

    def transform(self, x: ProjectedFloat) -> ProjectedFloat:
        assert 0.0 <= x <= 1.0, f'value {x} should be in [0, 1]'
        return np.log1p(self._scale * x) / self._output_size

    def reverse(self, x: ProjectedFloat) -> ProjectedFloat:
        assert 0.0 <= x <= 1.0, f'value {x} should be in [0, 1]'
        return np.expm1(x * self._output_size) / self._scale


OpenBound = t.Tuple[t.Optional[float], t.Optional[float]]
ConstraintSuggestion = t.Callable[[list], t.Dict[str, OpenBound]]


class Space:

    def __init__(
        self, *params: Param,
        constraints: t.List[t.Callable[[list], bool]] = None,
        constrained_bounds_suggestions: t.List[ConstraintSuggestion] = None,
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
        buffer = 'Space('
        for param in self.params:
            buffer += f'\n  {param!r},'
        buffer += '\n  constraints=['
        for constraint in self.constraints:
            buffer += f'\n    {constraint}'
        buffer += '],'
        buffer += '\n  constrained_bounds_suggestions=['
        for suggestion in self.constrained_bounds_suggestions:
            buffer += f'\n    {suggestion}'
        buffer += '],'
        buffer += '\n)'
        return buffer

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def sample(self, *, rng: RandomState) -> list:
        retries = 10
        bounds: t.Dict[str, OpenBound] = dict()

        for _ in range(retries):
            sample = []
            for param in self.params:
                lo, hi = bounds.get(param.name, (None, None))
                sample.append(param.sample(rng=rng, lo=lo, hi=hi))
            if all(c(sample) for c in self.constraints):
                return sample
            for suggestion in self.constrained_bounds_suggestions:
                for key, value in suggestion(sample).items():
                    if value is None:
                        continue
                    current_bounds = bounds.get(key, (None, None))
                    suggested_bounds = value
                    bounds[key] = merge_intervals(
                        current_bounds, suggested_bounds)

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
        return [p.mutate_transformed(x, rng=rng, relscale=s)
                for p, x, s in zip(self.params, sample_transformed, relscale)]

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


def merge_intervals(
    *intervals: OpenBound,
) -> OpenBound:

    def merge(
        left: t.Optional[float], right: t.Optional[float], *,
        operator: t.Callable,
    ) -> t.Optional[float]:
        if left is None:
            return right
        if right is None:
            return left
        return operator(left, right)

    merged_lo, merged_hi = None, None
    for interval in intervals:
        curr_lo, curr_hi = interval
        merged_lo = merge(merged_lo, curr_lo, operator=max)
        merged_hi = merge(merged_hi, curr_hi, operator=min)

    if merged_lo is not None and merged_hi is not None:
        assert merged_lo <= merged_hi

    return merged_lo, merged_hi

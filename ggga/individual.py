import numpy as np  # type: ignore


class Individual:
    def __init__(
        self, sample: list, *,
        fitness: float = None,
        gen: int = None,
        expected_improvement: float = None,
        prediction: float = None,
        cost: float = None,
    ) -> None:

        self._sample = sample

        self._cost = cost
        self._expected_improvement = expected_improvement
        self._fitness = fitness
        self._gen = gen
        self._prediction = prediction

    def __repr__(self):
        def default(optional_value, the_default):
            if optional_value is not None:
                return optional_value
            return the_default

        sample = ' '.join(repr(x) for x in self._sample)

        cost = default(self._cost, np.nan)
        expected_improvement = default(self._expected_improvement, np.nan)
        fitness = default(self._fitness, np.nan)
        gen = default(self._gen, np.nan)
        prediction = default(self._prediction, np.nan)

        return (f'Individual({fitness} @{cost:.2f} [{sample}]'
                f' prediction: {prediction}'
                f' ei: {expected_improvement}'
                f' gen: {gen})')

    @property
    def sample(self) -> list:
        return self._sample

    @property
    def fitness(self) -> float:
        assert self._fitness is not None
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        assert self._fitness is None
        self._fitness = value

    @property
    def cost(self) -> float:
        assert self._cost is not None
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        assert self._cost is None
        self._cost = value

    @property
    def gen(self) -> int:
        assert self._gen is not None
        return self._gen

    @gen.setter
    def gen(self, value: int) -> None:
        assert self._gen is None
        self._gen = value

    @property
    def expected_improvement(self) -> float:
        assert self._expected_improvement is not None
        return self._expected_improvement

    @expected_improvement.setter
    def expected_improvement(self, value: float) -> None:
        assert self._expected_improvement is None
        self._expected_improvement = value

    @property
    def prediction(self) -> float:
        assert self._prediction is not None
        return self._prediction

    @prediction.setter
    def prediction(self, value: float) -> None:
        assert self._prediction is None
        self._prediction = value

    def is_fully_initialized(self) -> bool:
        return all(field is not None for field in (
            self._cost,
            self._expected_improvement,
            self._fitness,
            self._gen,
            self._prediction,
        ))

import numpy as np  # type: ignore


class Individual:

    def __init__(
        self, sample: list, *,
        fitness: float = None,
        gen: int = None,
        ei: float = None,  # pylint: disable=invalid-name
        prediction: float = None,
        cost: float = None,
    ) -> None:

        self._sample = sample

        self._fitness = fitness
        self._gen = gen
        self._ei = ei
        self._prediction = prediction
        self._cost = cost

    def __repr__(self):
        def default(optional_value, the_default):
            if optional_value is not None:
                return optional_value
            return the_default

        fitness = default(self._fitness, np.nan)
        sample = ' '.join(repr(x) for x in self._sample)
        prediction = default(self._prediction, np.nan)
        ei = default(self._ei, np.nan)  # pylint: disable=invalid-name
        gen = default(self._gen, np.nan)
        cost = default(self._cost, np.nan)
        return (f'Individual({fitness} @{cost:.2f} [{sample}]'
                f' prediction: {prediction}'
                f' ei: {ei}'
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
    def ei(self) -> float:  # pylint: disable=invalid-name
        assert self._ei is not None
        return self._ei

    @ei.setter
    def ei(self, value: float) -> None:  # pylint: disable=invalid-name
        assert self._ei is None
        self._ei = value

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
            self._fitness, self._gen, self._ei, self._prediction, self._cost))

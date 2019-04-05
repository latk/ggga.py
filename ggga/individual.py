import numpy as np  # type: ignore


class Individual:
    """Parameters and result of a pending or completed experiment.

    Many fields are write-once.

    Parameters
    ----------
    sample: list
        The input variables at which the experiment shall be evaluated.
    """

    def __init__(
        self, sample: list, *,
        observation: float = None,
        gen: int = None,
        expected_improvement: float = None,
        prediction: float = None,
        cost: float = None,
    ) -> None:

        self._sample = sample

        self._cost = cost
        self._expected_improvement = expected_improvement
        self._observation = observation
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
        observation = default(self._observation, np.nan)
        gen = default(self._gen, np.nan)
        prediction = default(self._prediction, np.nan)

        return (f'Individual({observation} @{cost:.2f} [{sample}]'
                f' prediction: {prediction}'
                f' ei: {expected_improvement}'
                f' gen: {gen})')

    @property
    def sample(self) -> list:
        return self._sample

    @property
    def observation(self) -> float:
        """The observed value. Write-once."""
        assert self._observation is not None
        return self._observation

    @observation.setter
    def observation(self, value: float) -> None:
        assert self._observation is None
        self._observation = value

    @property
    def cost(self) -> float:
        """The observed cost. Write-once."""
        assert self._cost is not None
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        assert self._cost is None
        self._cost = value

    @property
    def gen(self) -> int:
        """The generation in which the Individual was evaluated. Write-once."""
        assert self._gen is not None
        return self._gen

    @gen.setter
    def gen(self, value: int) -> None:
        assert self._gen is None
        self._gen = value

    @property
    def expected_improvement(self) -> float:
        """The expected improvement before the Individual was evaluated.
        Write-once.
        """
        assert self._expected_improvement is not None
        return self._expected_improvement

    @expected_improvement.setter
    def expected_improvement(self, value: float) -> None:
        assert self._expected_improvement is None
        self._expected_improvement = value

    @property
    def prediction(self) -> float:
        """The predicted value. Write-once."""
        assert self._prediction is not None
        return self._prediction

    @prediction.setter
    def prediction(self, value: float) -> None:
        assert self._prediction is None
        self._prediction = value

    def is_fully_initialized(self) -> bool:
        """Check whether all write-once fields have been provided.
        If true, the object is immutable."""
        return all(field is not None for field in (
            self._cost,
            self._expected_improvement,
            self._observation,
            self._gen,
            self._prediction,
        ))

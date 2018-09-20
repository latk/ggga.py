import pytest  # type: ignore
import functools
import numpy as np  # type: ignore


def test_expected_improvement():
    from ..acquisition import expected_improvement

    assert expected_improvement(0, 1, 0) > expected_improvement(1, 1, 0), \
        "a lower mean should be more promising"
    assert expected_improvement(0, 2, -1) > expected_improvement(0, 1, -1), \
        "a larger std should be more promising"
    assert expected_improvement(0, 1, 0) > expected_improvement(0, 1, -1), \
        "a worse known minimum should be more promising"


def describe_gpr():
    from ..gpr import SurrogateModelGPR
    from ..space import Space, Real
    from numpy.random import RandomState  # type: ignore

    space = Space(Real('test', '--test', 0.0, 1.0))

    class SimpleModel:
        def __init__(self, model: SurrogateModelGPR) -> None:
            self.model: SurrogateModelGPR = model

        def predict(self, x: float) -> float:
            return self.model.predict([x], return_std=False)

        def uncertainty(self, x: float) -> float:
            y, std = self.model.predict([x])
            return std

    def describe_differing_sample_density():

        @pytest.fixture(scope='module')
        def model():
            xs = [0.1, 0.5, 0.5, 0.9]
            ys = [1.0, 1.8, 2.2, 3.0]
            model = SurrogateModelGPR.estimate(
                [[x] for x in xs], ys,
                space=space,
                rng=RandomState(123),
                prior=None,
            )
            return SimpleModel(model)

        def it_should_roughly_fit_the_data(model):
            assert model.predict(0.1) == pytest.approx(1, abs=0.1)
            assert model.predict(0.5) == pytest.approx(2, abs=0.1)
            assert model.predict(0.9) == pytest.approx(3, abs=0.1)

        def it_should_provide_a_reasonable_interpolation(model):
            assert model.predict(0.3) == pytest.approx(1.5, abs=0.2)
            assert model.predict(0.7) == pytest.approx(2.5, abs=0.2)

        def it_should_prefer_a_conservative_extrapolation(model):
            assert model.predict(0.0) == pytest.approx(0.9, abs=0.1)
            assert model.predict(1.0) == pytest.approx(3.1, abs=0.1)

        def it_should_have_similar_uncertainty_for_single_observations(model):
            assert model.uncertainty(0.1) \
                == pytest.approx(model.uncertainty(0.9))

        def it_should_have_lower_uncertainty_for_more_observations(model):
            assert model.uncertainty(0.5) < model.uncertainty(0.1)

    def describe_unsampled_regions():

        @pytest.fixture(scope='module')
        def model():
            xs = [0.3, 0.5, 0.7]
            ys = [1.0, 2.0, 1.5]
            model = SurrogateModelGPR.estimate(
                [[x] for x in xs], ys,
                space=space,
                rng=RandomState(42),
                prior=None,
            )
            return SimpleModel(model)

        def it_has_low_uncertainty_at_samples(model):
            assert model.uncertainty(0.3) < 0.01
            assert model.uncertainty(0.5) < 0.01
            assert model.uncertainty(0.7) < 0.01

        def it_should_have_more_uncertainty_for_interpolation(model):
            assert model.uncertainty(0.4) > 10 * model.uncertainty(0.3)
            assert model.uncertainty(0.6) > 10 * model.uncertainty(0.3)

        def it_should_have_more_uncertainty_for_extrapolation(model):
            assert model.uncertainty(0.0) > 10 * model.uncertainty(0.3)
            assert model.uncertainty(1.0) > 10 * model.uncertainty(0.3)

def describe_params():

    def describe_integers():
        from ..space import Integer

        def describe_sample():

            def it_produces_uniform_samples():
                lo, hi = 0, 9
                rng = np.random.RandomState(1717)
                param = Integer('test', '--test', lo, hi)

                counts = [0] * 10
                for _ in range(30):
                    sample = param.sample(rng=rng)
                    counts[sample] += 1

                assert counts == pytest.approx([3] * 10, abs=2)

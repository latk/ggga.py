import warnings
import typing as t

import attr
import pytest  # type: ignore
import numpy as np  # type: ignore
import skopt  # type: ignore
from ..util import fork_random_state


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

        def __repr__(self) -> str:
            return f"SimpleModel({self.model.kernel!r})"

        def predict(self, x: float) -> float:
            y, _ = self.model.predict([x], return_std=False)
            return y

        def uncertainty(self, x: float) -> float:
            _, std = self.model.predict([x])
            assert std is not None
            return std

    def describe_differing_sample_density():

        @pytest.fixture(scope='module')
        def model():  # pylint: disable=unused-variable
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
            xs = [0.1, 0.5, 0.9]
            expected_ys = [1.0, 2.0, 3.0]
            predicted_ys = [model.predict(x) for x in xs]

            assert predicted_ys == pytest.approx(expected_ys, abs=0.1)

        def it_should_provide_a_reasonable_interpolation(model):
            assert model.predict(0.3) == pytest.approx(1.5, abs=0.2)
            assert model.predict(0.7) == pytest.approx(2.5, abs=0.2)

        def it_should_prefer_a_conservative_extrapolation(model):
            assert model.predict(0.0) == pytest.approx(0.9, abs=0.1)
            assert model.predict(1.0) == pytest.approx(3.1, abs=0.1)

        def it_should_have_similar_uncertainty_for_single_observations(model):
            uncertainty_0_1 = model.uncertainty(0.1)
            uncertainty_0_9 = model.uncertainty(0.9)
            assert uncertainty_0_1 == pytest.approx(uncertainty_0_9)

        def it_should_have_lower_uncertainty_for_more_observations(model):
            assert model.uncertainty(0.5) < model.uncertainty(0.1)

    def describe_unsampled_regions():

        @pytest.fixture(scope='module')
        def model():  # pylint: disable=unused-variable
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

    def it_works_in_1d():
        from ..benchmark_functions import sphere

        xs = np.linspace(-2.0, 2.0, 5).reshape(-1, 1)
        ys = sphere(xs.reshape(-1))
        assert np.all(ys == np.array([4.0, 1.0, 0.0, 1.0, 4.0]))

        space = Space(Real('x', '-x', -2, 2))
        model = SurrogateModelGPR.estimate(
            xs, ys,
            space=space, rng=RandomState(123), prior=None,
            noise_bounds=(1e-2, 1e1),
            length_scale_bounds=(1e-2, 1e1),
        )

        def check_predictions(xs, extra_std=0):
            expected_ys = sphere(xs.reshape(-1))

            predicted_ys, predicted_std = model.predict_a(xs)
            lower_bound = expected_ys - 1.0 * predicted_std - extra_std
            upper_bound = expected_ys + 0.6 * predicted_std + extra_std
            locs = ~(lower_bound <= predicted_ys)
            locs |= ~(predicted_ys <= upper_bound)
            locs |= ~(predicted_std <= extra_std + 0.1)
            if np.any(locs):
                raise AssertionError(
                    f"Prediction failed:"
                    f"\nxs            = {xs[locs].reshape(-1)}"
                    f"\npredicted ys  = {predicted_ys[locs]}"
                    f"\nexpected  ys  = {expected_ys[locs]}"
                    f"\npredicted std = {predicted_std[locs]}"
                    f"\nexpected  std = {extra_std}")

        check_predictions(xs, extra_std=0.2)
        check_predictions(
            np.array([-1.5, -0.5, 0.5, 1.5]).reshape(-1, 1),
            extra_std=0.3)

    @pytest.fixture(params=['rng123', 'rng171718', 'rng6657'])
    def rng(request):  # pylint: disable=unused-variable
        return np.random.RandomState(int(request.param[3:]))

    @pytest.fixture(params=['gridTraining', 'randomTraining'])
    def training_data(  # pylint: disable=unused-variable
        request, rng: RandomState, noise_level: float,
    ):
        from ..benchmark_functions import sphere

        if request.param == 'randomTraining':
            xs = rng.rand(50, 2) * 4 - 2
        elif request.param == 'gridTraining':
            gridaxis = np.linspace(-2, 2, 7)
            xs = np.dstack(np.meshgrid(gridaxis, gridaxis)).reshape(-1, 2)
        else:
            raise NotImplementedError(request.param)

        ys = rng.normal(sphere(*xs.T), noise_level)

        n_observations = len(xs)
        assert xs.shape == (n_observations, 2)
        assert ys.shape == (n_observations,)

        space = Space(
            Real('x', '-x', -2, 2),
            Real('y', '-y', -2, 2),
        )
        model = SurrogateModelGPR.estimate(
            xs, ys,
            space=space, rng=rng, prior=None,
            noise_bounds=(1e-2, 1e1),
            length_scale_bounds=(1e-2, 2e1),
        )

        skopt_model = skopt.learning.GaussianProcessRegressor(
            kernel=(
                skopt.learning.gaussian_process.kernels.sk_ConstantKernel()
                * skopt.learning.gaussian_process.kernels.Matern(nu=5/2)
                + skopt.learning.gaussian_process.kernels.WhiteKernel()),
            n_restarts_optimizer=2,
            random_state=fork_random_state(rng),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message="^fmin_l_bfgs_b terminated abnormally")
            skopt_model.fit(xs, ys)

        return xs, ys, model, skopt_model, request.param

    @pytest.fixture(params=['noNoise', 'lowNoise', 'mediumNoise', 'highNoise'])
    def noise_level(request):  # pylint: disable=unused-variable
        return {
            'noNoise': 0.0,
            'lowNoise': 0.1,
            'mediumNoise': 1.0,
            'highNoise': 4.0,
        }[request.param]

    @attr.s(auto_attribs=True)
    class Case:
        xs: np.ndarray = attr.ib()
        extra_std: float = 0.0
        allowed_failures: int = 0
        training_ys: t.Optional[np.ndarray] = None

    @pytest.fixture(params=['selftest', 'newsample'])
    def case(  # pylint: disable=unused-variable
        request, training_data, noise_level, rng,
    ):
        xs, ys, _model, _skopt_model, training_type = training_data

        allowed_failures = 0
        allowed_noise = noise_level + 0.2
        if noise_level >= 1.0:
            allowed_failures += 1
        if training_type == 'randomTraining':
            allowed_noise += 0.1
        if request.param == 'newsample':
            allowed_failures += 1
        if (training_type, request.param) == ('randomTraining', 'newsample'):
            allowed_failures += 1

        if request.param == 'selftest':
            return Case(xs, allowed_noise,
                        allowed_failures=allowed_failures,
                        training_ys=ys)

        if request.param == 'newsample':
            xs_test = np.vstack((
                rng.rand(15, 2) * 4 - 2,
                rng.rand(10, 2) * 2 - 1,
            ))

            return Case(xs_test, allowed_noise + 0.1,
                        allowed_failures=allowed_failures)

        raise NotImplementedError(request.param)

    def it_works_2d(  # for test name: pylint: disable=unused-argument
        training_data, case, noise_level, rng,
    ):
        from ..benchmark_functions import sphere

        _xs, _ys, model, skopt_model, _ = training_data

        test_xs = case.xs
        extra_std = case.extra_std
        allowed_failures = case.allowed_failures
        training_ys = case.training_ys

        expected_ys = sphere(*test_xs.T)
        predicted_ys, predicted_std = model.predict_a(test_xs)

        mse_ys = np.mean((predicted_ys - expected_ys)**2)
        assert mse_ys < 2 * extra_std

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="^Predicted variances smaller than 0.")
            skopt_predicted_ys, skopt_predicted_std = \
                skopt_model.predict(test_xs, return_std=True)

        def loc_info(locs: np.ndarray, *, models):
            yield 'x1       ', test_xs[locs, 0]
            yield 'x2       ', test_xs[locs, 1]

            yield 'ys our   ', predicted_ys[locs]
            yield 'ys skopt ', skopt_predicted_ys[locs]
            yield 'ys expec ', expected_ys[locs]
            if training_ys is not None:
                yield 'ys train ', training_ys[locs]

            yield 'std our  ', predicted_std[locs]
            yield 'std skopt', skopt_predicted_std[locs]

            if models:
                model_items = model.kernel.get_params()
                skopt_items = skopt_model.kernel_.get_params()
                for key in sorted(set(model_items) & set(skopt_items)):
                    yield f"model {key}", model_items.pop(key)
                    yield f"skopt {key}", skopt_items.pop(key)
                yield from (
                    (f"model {key}", value) for key, value
                    in sorted(model_items.items()))
                yield from (
                    (f"skopt {key}", value) for key, value
                    in sorted(skopt_items.items()))

        def fmt_loc_info(
            msg: str, locs: np.ndarray, *, models: bool = False,
        ) -> str:
            return msg + ''.join(f"\n{k} = {v}" for k, v
                                 in loc_info(locs, models=models))

        def bound(zscore):
            bounds = expected_ys + zscore * predicted_std
            bounds += np.sign(zscore) * extra_std
            if training_ys is not None:
                if zscore < 0:
                    bounds = np.min([bounds, training_ys], axis=0)
                elif zscore > 0:
                    bounds = np.max([bounds, training_ys], axis=0)
            return bounds

        prediction_ok = bound(-2.0) <= predicted_ys
        prediction_ok &= predicted_ys <= bound(1.0)
        if np.sum(~prediction_ok) > allowed_failures:
            raise AssertionError(fmt_loc_info(
                "Incorrect prediction:", ~prediction_ok))

        std_ok = predicted_std < 1.5 * extra_std
        if np.sum(~std_ok) > allowed_failures:
            raise AssertionError(fmt_loc_info(
                "Detected large variances:", ~std_ok))

        effective_std = np.max([predicted_std, skopt_predicted_std], axis=0)
        prediction_delta_ys = np.abs(predicted_ys - skopt_predicted_ys)
        skopt_length_scale = [
            value
            for key, value in skopt_model.kernel_.get_params().items()
            if key.endswith('__length_scale')
        ][0]
        # Check if the predictions match skopt,
        skopt_ok = prediction_delta_ys < effective_std + extra_std / 3
        # ... but only if skopt is noticeably better than our predictions
        skopt_ok |= ~(np.abs(skopt_predicted_ys - expected_ys) + 0.2
                      < np.abs(predicted_ys - expected_ys))
        # ... and only if skopt did not overfit by choosing a tiny length scale
        skopt_ok |= ~(skopt_length_scale > 0.05)
        if np.sum(~skopt_ok) > allowed_failures:
            raise AssertionError(fmt_loc_info(
                "Prediction does not match skopt:", ~skopt_ok,
                models=True))

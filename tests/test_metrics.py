import numpy as np
import pytest

from koopman.metrics import error_curve, rmse, state_scale, valid_prediction_time


def test_rmse():
    a = np.zeros((4, 3))
    assert rmse(a, a) == 0.0
    assert rmse(a, a + 2.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        rmse(a, np.zeros((3, 3)))


def test_state_scale_is_rms_distance_from_the_mean():
    ref = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert state_scale(ref) == pytest.approx(1.0)
    assert state_scale(np.ones((10, 3))) == pytest.approx(0.0)
    assert state_scale(np.stack([ref, ref])) == pytest.approx(1.0)  # accepts (n_traj, T, d)


def test_error_curve_values_and_shapes():
    true = np.zeros((2, 5, 3))
    pred = np.zeros((2, 5, 3))
    pred[..., 0] = 3.0
    pred[..., 1] = 4.0
    c = error_curve(true, pred, scale=5.0)
    assert c.shape == (2, 5)
    assert np.allclose(c, 1.0)
    with pytest.raises(ValueError):
        error_curve(true, pred[:, :4])


def test_valid_prediction_time_crossing():
    t = np.arange(10.0)
    curve = np.array([0, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 1, 1, 1])
    vt, censored = valid_prediction_time(curve, t, threshold=0.2)
    assert vt == 4.0 and censored is False


def test_valid_prediction_time_is_censored_if_threshold_never_crossed():
    t = np.arange(5.0)
    vt, censored = valid_prediction_time(np.zeros(5), t, threshold=0.2)
    assert vt == 4.0 and censored is True


def test_valid_prediction_time_shape_check():
    with pytest.raises(ValueError):
        valid_prediction_time(np.zeros(4), np.arange(5.0))

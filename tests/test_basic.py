import numpy as np
import pytest

from pkoffee.metrics import compute_rmse
from pkoffee.parametric_function import Quadratic


def test_compute_rmse_perfect_fit_is_zero():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert compute_rmse(y_true, y_pred) == 0.0


def test_compute_rmse_small_error():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_pred = np.array([1.1, 1.9, 3.1, 3.9], dtype=np.float32)

    assert compute_rmse(y_true, y_pred) == pytest.approx(0.1, rel=1e-6)


def test_quadratic_evaluation():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    q = Quadratic()

    result = q(x, 1.0, 2.0, 3.0)
    expected = np.array([1.0, 6.0, 17.0], dtype=np.float32)

    assert np.allclose(result, expected)

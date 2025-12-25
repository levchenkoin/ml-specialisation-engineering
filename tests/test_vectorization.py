import numpy as np
import pytest

from ml_linear_regression_uni.vectorization import my_dot, dot_np, as_vector


def test_as_vector_makes_1d():
    x = np.array([[1.0], [2.0], [3.0]])
    v = as_vector(x)
    assert v.shape == (3,)
    assert np.allclose(v, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        (np.array([1.0, -2.0, 0.5]), np.array([3.0, 4.0, -1.0])),
        (np.arange(10), np.arange(10)[::-1]),
    ],
)
def test_my_dot_matches_numpy(a, b):
    assert np.isclose(my_dot(a, b), dot_np(a, b))


def test_dot_raises_on_shape_mismatch():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        my_dot(a, b)
    with pytest.raises(ValueError):
        dot_np(a, b)

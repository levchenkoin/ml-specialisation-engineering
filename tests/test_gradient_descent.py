import numpy as np
from ml_linear_regression_uni.linear import gradient_descent, cost


def test_gradient_descent_monotonic_nonincreasing_on_simple_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 50)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.1, size=x.shape)
    w, b, hist = gradient_descent(x, y, w0=0.0, b0=0.0, alpha=0.05, epochs=200)
    assert hist[-1] < hist[0]

    non_increasing_steps = sum(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))
    assert non_increasing_steps / (
                len(hist) - 1) > 0.8
    assert cost(x, y, w, b) < cost(x, y, 0.0, 0.0)

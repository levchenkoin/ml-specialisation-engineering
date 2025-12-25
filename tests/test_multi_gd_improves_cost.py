import numpy as np
from ml_linear_regression_uni.multi import gradient_descent, cost


def test_gd_decreases_cost_on_synthetic_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    w_true = np.array([2.0, -1.0, 0.5, 3.0])
    y = X @ w_true + 0.2 + rng.normal(0, 0.05, size=80)

    w0 = np.zeros(4)
    b0 = 0.0
    j0 = cost(X, y, w0, b0)

    w, b, hist = gradient_descent(X, y, w0=w0, b0=b0, alpha=0.05, epochs=300)
    assert hist[-1] < j0
    assert cost(X, y, w, b) < j0

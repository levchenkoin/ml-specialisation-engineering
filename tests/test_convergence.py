# tests/test_convergence.py
import numpy as np
from ml_linear_regression_uni.linear import fit_least_squares, gradient_descent, cost

def test_gd_converges_close_to_closed_form():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 60)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.15, x.shape)

    w_ls, b_ls = fit_least_squares(x, y)
    w_gd, b_gd, _ = gradient_descent(x, y, w0=0.0, b0=0.0, alpha=0.05, epochs=800)

    assert abs(w_gd - w_ls) < 0.05
    assert abs(b_gd - b_ls) < 0.2
    assert cost(x, y, w_gd, b_gd) <= cost(x, y, 0.0, 0.0)
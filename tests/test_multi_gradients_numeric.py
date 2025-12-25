import numpy as np
from ml_linear_regression_uni.multi import cost, gradients


def test_multi_gradients_match_numeric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    true_w = np.array([1.5, -2.0, 0.7])
    y = X @ true_w + 0.3 + rng.normal(0, 0.01, size=20)

    w = rng.normal(size=3)
    b = 0.1

    dj_dw, dj_db = gradients(X, y, w, b)

    eps = 1e-6
    # numeric for b
    j1 = cost(X, y, w, b + eps)
    j2 = cost(X, y, w, b - eps)
    dj_db_num = (j1 - j2) / (2 * eps)
    assert abs(dj_db - dj_db_num) < 1e-4

    # numeric for w
    for k in range(w.size):
        w1 = w.copy(); w1[k] += eps
        w2 = w.copy(); w2[k] -= eps
        dj_dw_num = (cost(X, y, w1, b) - cost(X, y, w2, b)) / (2 * eps)
        assert abs(dj_dw[k] - dj_dw_num) < 1e-4

import numpy as np
from ml_linear_regression_uni.linear import cost, gradients


def numeric_gradients(x, y, w, b, eps=1e-6):
    # dJ/dw ~ (J(w+eps,b) - J(w-eps,b)) / (2eps)
    dw = (cost(x, y, w + eps, b) - cost(x, y, w - eps, b)) / (2 * eps)
    db = (cost(x, y, w, b + eps) - cost(x, y, w, b - eps)) / (2 * eps)
    return dw, db


def test_gradients_match_numeric():
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = 2.5 * x - 0.7 + rng.normal(scale=0.05, size=x.shape)

    w, b = 0.3, -0.2
    dj_dw, dj_db = gradients(x, y, w, b)
    ndw, ndb = numeric_gradients(x, y, w, b)

    assert abs(dj_dw - ndw) < 1e-4
    assert abs(dj_db - ndb) < 1e-4

import numpy as np
from ml_linear_regression_uni.multi import predict, cost


def test_predict_shape_and_cost_is_scalar():
    X = np.array([[1, 2],
                  [3, 4]], dtype=float)
    w = np.array([10, 20], dtype=float)
    b = 5.0

    y_pred = predict(X, w, b)
    assert y_pred.shape == (2,)
    j = cost(X, np.array([0, 0], dtype=float), w, b)
    assert isinstance(j, float)

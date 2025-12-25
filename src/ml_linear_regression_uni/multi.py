from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    X: (m, n), w: (n,), b: scalar
    returns: (m,)
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    return X @ w + float(b)


def cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert X.shape[0] == y.shape[0], "X и y должны иметь одинаковое число примеров (m)"
    m = X.shape[0]
    residuals = predict(X, w, b) - y
    return float((residuals @ residuals) / (2 * m))


def gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """
    dj_dw: (n,), dj_db: scalar
    Формулы:
      err = (Xw + b - y)  -> (m,)
      dj_dw = (1/m) * X^T err
      dj_db = (1/m) * sum(err)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    m = X.shape[0]
    err = (X @ w + float(b)) - y  # (m,)
    dj_dw = (X.T @ err) / m       # (n,)
    dj_db = float(err.sum() / m)  # scalar
    return dj_dw, dj_db


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    b0: float,
    alpha: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    w = np.asarray(w0, dtype=float).reshape(-1).copy()
    b = float(b0)
    history: list[float] = []

    for _ in range(int(epochs)):
        # сохраняем cost ДО обновления или ПОСЛЕ — ок; главное единообразно
        history.append(cost(X, y, w, b))
        dj_dw, dj_db = gradients(X, y, w, b)
        w -= float(alpha) * dj_dw
        b -= float(alpha) * dj_db

    history.append(cost(X, y, w, b))
    return w, b, history


@dataclass
class LinearRegressorMulti:
    w: np.ndarray
    b: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict(X, self.w, self.b)

    @classmethod
    def from_gradient_descent(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 5e-7,
        epochs: int = 1000,
        w0: np.ndarray | None = None,
        b0: float = 0.0,
    ) -> "LinearRegressorMulti":
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        if w0 is None:
            w0 = np.zeros(n, dtype=float)
        w, b, _ = gradient_descent(X, y, w0=w0, b0=b0, alpha=alpha, epochs=epochs)
        return cls(w=w, b=b)

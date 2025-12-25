from dataclasses import dataclass
import numpy as np

def f_wb(x: np.ndarray, w: float, b: float) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return w * x + b

def cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape == y.shape, "x и y должны быть одинаковой длины"
    m = x.shape[0]
    residuals = f_wb(x, w, b) - y
    return float((residuals @ residuals) / (2 * m))

def fit_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Возвращает (w, b) по формуле МНК для одномерной регрессии.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape == y.shape and x.size >= 2
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    # w = sum((x - x̄)(y - ȳ)) / sum((x - x̄)^2)
    num = float(((x - x_mean) * (y - y_mean)).sum())
    den = float(((x - x_mean) ** 2).sum())
    w = num / den if den != 0 else 0.0
    b = y_mean - w * x_mean
    return w, b

def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """
    ∂J/∂w = (1/m) * sum( (w x_i + b - y_i) * x_i )
    ∂J/∂b = (1/m) * sum( w x_i + b - y_i )
    """
    x = np.asarray(x).reshape(-1); y = np.asarray(y).reshape(-1)
    m = x.shape[0]
    pred = f_wb(x, w, b)
    err = pred - y
    dj_dw = float((err * x).sum() / m)
    dj_db = float(err.sum() / m)
    return dj_dw, dj_db

def gradient_descent(
    x: np.ndarray, y: np.ndarray, w0: float, b0: float, alpha: float, epochs: int
) -> tuple[float, float, list[float]]:
    """
    Возвращает (w, b, history), где history — список значений стоимости по эпохам.
    """
    w, b = float(w0), float(b0)
    history: list[float] = []
    for _ in range(epochs):
        dj_dw, dj_db = gradients(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        history.append(cost(x, y, w, b))

    return w, b, history

@dataclass
class LinearRegressorUni:
    w: float = 0.0
    b: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return f_wb(x, self.w, self.b)

    @classmethod
    def from_two_points(cls, x: np.ndarray, y: np.ndarray) -> "LinearRegressorUni":
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        assert x.size == 2 and y.size == 2, "Нужно две точки"
        w = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - w * x[0]
        return cls(float(w), float(b))

    @classmethod
    def from_least_squares(cls, x: np.ndarray, y: np.ndarray) -> "LinearRegressorUni":
        w, b = fit_least_squares(x, y)
        return cls(w=float(w), b=float(b))

import argparse
import numpy as np
from ml_linear_regression_uni.multi import gradient_descent, cost


def main() -> None:
    p = argparse.ArgumentParser(description="Multivariate linear regression (GD)")
    p.add_argument("--alpha", type=float, default=5e-7)
    p.add_argument("--epochs", type=int, default=1000)

    args = p.parse_args()

    # Данные из лабы (как baseline)
    X = np.array([[2104, 5, 1, 45],
                  [1416, 3, 2, 40],
                  [852,  2, 1, 35]], dtype=float)
    y = np.array([460, 232, 178], dtype=float)

    w0 = np.zeros(X.shape[1], dtype=float)
    b0 = 0.0

    w, b, hist = gradient_descent(X, y, w0=w0, b0=b0, alpha=args.alpha, epochs=args.epochs)
    print(f"Final cost: {cost(X, y, w, b):.6f}")
    print(f"b={b:.6f}")
    print(f"w={w}")


if __name__ == "__main__":
    main()

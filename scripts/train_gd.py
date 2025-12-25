import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ml_linear_regression_uni.linear import gradient_descent, cost

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", nargs="+", type=float, default=[1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    p.add_argument("--y", nargs="+", type=float, default=[250, 300, 480, 430, 630, 730])
    p.add_argument("--w0", type=float, default=0.0)
    p.add_argument("--b0", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--out", type=str, default="reports/cost_vs_epochs_gd.png")
    args = p.parse_args()

    x = np.array(args.x, dtype=float)
    y = np.array(args.y, dtype=float)

    w, b, hist = gradient_descent(x, y, args.w0, args.b0, args.alpha, args.epochs)
    print(f"Final w={w:.3f}, b={b:.3f}, cost={cost(x,y,w,b):.3f}")

    # Кривая стоимости
    plt.plot(hist)
    plt.title("Cost over epochs (gradient descent)")
    plt.xlabel("epoch")
    plt.ylabel("J(w,b)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
import argparse

from ml_linear_regression_uni.vectorization import benchmark_dot


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Python loop dot vs NumPy dot")
    p.add_argument("--n", type=int, default=1_000_000, help="vector size")
    p.add_argument("--repeats", type=int, default=3, help="number of repeats")
    args = p.parse_args()

    r = benchmark_dot(n=args.n, repeats=args.repeats)
    print(f"n={r.n}, repeats={r.repeats}")
    print(f"my_dot:  {r.loop_ms:.2f} ms")
    print(f"np.dot:  {r.numpy_ms:.2f} ms")
    print(f"speedup: {r.speedup:.1f}x")


if __name__ == "__main__":
    main()

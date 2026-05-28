import numpy as np
from pathlib import Path

BUILD_DIR = Path("build")

def load_csv(filename):
    return np.loadtxt(BUILD_DIR / filename, delimiter=",", dtype=np.float32)

def main():
    A = load_csv("A_128.csv")
    B = load_csv("B_128.csv")
    C_cpp = load_csv("C_128_cpp.csv")
    R_cpp = load_csv("R_128_cpp.csv")
    S_cpp = load_csv("S_128_cpp.csv").flatten()

    C_numpy = A @ B
    R_numpy = np.maximum(C_numpy, 0)

    C_flat = C_numpy.flatten()
    S_numpy = np.exp(C_flat - np.max(C_flat))
    S_numpy = S_numpy / np.sum(S_numpy)

    matmul_abs_error = np.max(np.abs(C_numpy - C_cpp))
    matmul_rel_error = np.max(
        np.abs(C_numpy - C_cpp) / np.maximum(np.abs(C_numpy), 1e-8)
    )

    relu_abs_error = np.max(np.abs(R_numpy - R_cpp))
    relu_rel_error = np.max(
        np.abs(R_numpy - R_cpp) / np.maximum(np.abs(R_numpy), 1e-8)
    )

    softmax_abs_error = np.max(np.abs(S_numpy - S_cpp))
    softmax_rel_error = np.max(
        np.abs(S_numpy - S_cpp) / np.maximum(np.abs(S_numpy), 1e-8)
    )

    print("Floating-Point Accuracy Validation")
    print("----------------------------------")
    print(f"Matrix multiplication max absolute error: {matmul_abs_error:.8f}")
    print(f"Matrix multiplication max relative error: {matmul_rel_error:.8f}")
    print(f"ReLU max absolute error: {relu_abs_error:.8f}")
    print(f"ReLU max relative error: {relu_rel_error:.8f}")
    print(f"Softmax max absolute error: {softmax_abs_error:.8f}")
    print(f"Softmax max relative error: {softmax_rel_error:.8f}")

    tolerance = 1e-3

    if (matmul_abs_error < tolerance and relu_abs_error < tolerance and softmax_abs_error < tolerance):
        print("\nValidation PASSED.")
    else:
        print("\nValidation WARNING: Error exceeded tolerance.")

if __name__ == "__main__":
    main()
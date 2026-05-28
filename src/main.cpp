#include "matrix.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> sizes = {64, 128, 256, 512};

    std::cout << "AI Numeric Kernel Benchmark\n";
    std::cout << "Kernel: CPU Matrix Multiplication + ReLU\n\n";

    for (int size : sizes) {
        int M = size;
        int N = size;
        int K = size;

        std::vector<float> A = generate_matrix(M, K);
        std::vector<float> B = generate_matrix(K, N);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float> C = matmul_cpu(A, B, M, N, K);
        std::vector<float> R = relu_cpu(C);
        std::vector<float> S = softmax_cpu(C);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Size: " << size << "x" << size
                  << " | Runtime: " << elapsed.count() << " ms"
                  << " | Output elements: " << R.size()
                  << "\n";

        if (size == 128) {
            save_matrix("A_128.csv", A, M, K);
            save_matrix("B_128.csv", B, K, N);
            save_matrix("C_128_cpp.csv", C, M, N);
            save_matrix("R_128_cpp.csv", R, M, N);
            save_matrix("S_128_cpp.csv", S, 1, static_cast<int>(S.size()));
        }
    }

    std::cout << "\nSaved 128x128 outputs for NumPy validation.\n";

    return 0;
}
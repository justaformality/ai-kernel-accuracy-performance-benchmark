#include "matrix.hpp"

#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

std::vector<float> generate_matrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& value : matrix) {
        value = distribution(generator);
    }

    return matrix;
}

std::vector<float> matmul_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int N,
    int K
) {
    if (static_cast<int>(A.size()) != M * K) {
        throw std::invalid_argument("Matrix A has incorrect size.");
    }

    if (static_cast<int>(B.size()) != K * N) {
        throw std::invalid_argument("Matrix B has incorrect size.");
    }

    std::vector<float> C(M * N, 0.0f);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }

    return C;
}

std::vector<float> relu_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

    return output;
}

void save_matrix(
    const std::string& filename,
    const std::vector<float>& matrix,
    int rows,
    int cols
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing.");
    }

    file << std::fixed << std::setprecision(8);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << matrix[i * cols + j];

            if (j < cols - 1) {
                file << ",";
            }
        }

        file << "\n";
    }
}
std::vector<float> softmax_cpu(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    float max_value = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }

    for (size_t i = 0; i < output.size(); i++) {
        output[i] /= sum;
    }

    return output;
}
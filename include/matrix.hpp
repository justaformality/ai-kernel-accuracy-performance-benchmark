#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>

std::vector<float> generate_matrix(int rows, int cols);

std::vector<float> matmul_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int N,
    int K
);

std::vector<float> relu_cpu(const std::vector<float>& input);

void save_matrix(
    const std::string& filename,
    const std::vector<float>& matrix,
    int rows,
    int cols
);

std::vector<float> softmax_cpu(const std::vector<float>& input);

#endif
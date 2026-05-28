**AI KERNEL ACCURACY & PERFORMANCE BENCHMARK**
This project is a small C++ benchmark framework I built to better understand how AI numerical operations work under the hood. It focuses on implementing core AI-style kernels from scratch, validating floating point accuracy, and benchmarking runtime performance.

The project currently includes:
- Matrix multiplication
- ReLU activation
- Numerically stable softmax
- NumPy based validation
- CPU runtime benchmarking

**Why I Built This**
A lot of AI frameworks rely heavily on optimized numerical kernels behind the scenes, especially operations like matrix multiplication and activation functions.

Instead of just using existing libraries, I wanted to build some of these operations myself in C++ and understand:
- how floating point computations behave
- how numerical validation works
- how softmax stability is handled
- how runtime changes with larger workloads
- how AI-style kernels are structured internally

**Tech Stack**
C++17
Python
NumPy
CMake
Command line development

**Features**
- Matrix muliplication implemented from scratch in C++
- ReLU activation kernel
- Numerically stable softmax implementation
- Floating point validation against NumPy reference outputs
- Absolute and relative error checking
- Runtime benchmarking across different matrix sizes
- Modular C++ project structure using CMake

**Project Structure**
include/
  matrix.hpp

src/
  main.cpp
  matrix_cpu.cpp

tests/
  validate.py

benchmarks/
  results.md

build/

**How to Build**
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

**Run the Benchmark**
./ai_benchmark

EXAMPLE OUTPUT:
AI Numeric Kernel Benchmark
Kernel: CPU Matrix Multiplication + ReLU

Size: 64x64 | Runtime: 3.06977 ms | Output elements: 4096
Size: 128x128 | Runtime: 20.5759 ms | Output elements: 16384
Size: 256x256 | Runtime: 83.3207 ms | Output elements: 65536
Size: 512x512 | Runtime: 746.832 ms | Output elements: 262144

Saved 128x128 outputs for NumPy validation.

**Validate Numerical Accuracy**
From the project root:

python3 tests/validate.py

EXAMPLE VALIDATION OUTPUT:
Floating-Point Accuracy Validation
----------------------------------
Matrix multiplication max absolute error: 0.00000763
Matrix multiplication max relative error: 0.01291928
ReLU max absolute error: 0.00000668
ReLU max relative error: 0.00580442
Softmax max absolute error: 0.00002599
Softmax max relative error: 0.49994466

Validation PASSED.

**Benchmarking**
The benchmark currently tests runtime performance for:
- 64x64 matrices
- 128x128 matrices
- 256x256 matrices
- 512x512 matrices

This helps analyze how computation time scales as matrix size increases

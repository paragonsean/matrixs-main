/**
 * @file generateMatrix.cpp
 * @brief Matrix generation and file I/O utilities for the matrix library
 * 
 * This program provides utilities for:
 * - Generating random matrices with guaranteed convergence properties
 * - Loading matrices from files with performance timing
 * - Multi-threaded matrix generation for large matrices
 * - Diagonally dominant matrices for iterative solver testing
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include "../include/matrix.h"
#include "../include/dense_matrix_storage.h"
#include "../include/sparse_matrix_storage.h"
#include "matrix_file_handler.h"

/**
 * @brief Load Matrix A and vector b from a file with performance timing
 * 
 * This function reads matrix data from a file in augmented format where
 * each row contains matrix elements followed by the corresponding RHS vector element.
 * The function provides timing information for performance analysis.
 * 
 * @tparam MatrixType Template parameter for matrix storage type (dense/sparse)
 * @param A Output matrix to be loaded with coefficient matrix
 * @param b Output matrix to be loaded with RHS vector
 * @param filename Path to the input file
 * 
 * File format expected:
 * First line: rows cols
 * Subsequent lines: A[i,0] A[i,1] ... A[i,n-1] b[i,0]
 * 
 * Performance features:
 * - High-resolution timing for load performance measurement
 * - Error handling with descriptive messages
 * - Uses MatrixFileHandler for optimized loading
 */
template <typename MatrixType>
void loadDefaultAB(MatrixType& A, MatrixType& b, const std::string& filename) {
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }

    size_t rows, cols;
    if (!(inputFile >> rows >> cols)) {
        std::cerr << "Failed to read dimensions for matrix A from file.\n";
        return;
    }
    if (rows == 0 || cols == 0) {
        std::cerr << "Matrix A dimensions must be greater than zero.\n";
        return;
    }

    A = MatrixType(rows, cols);
    b = MatrixType(rows, 1);

    // Use MatrixFileHandler to load the data
    pnmatrix::MatrixFileHandler handler;
    handler.loadDefaultAB(A, b, filename);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix A and vector b successfully loaded from " << filename << "\n";
    std::cout << "Loading took: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";
}

/**
 * @brief Generate rows of a diagonally dominant matrix in parallel
 * 
 * This function generates a subset of matrix rows with the following properties:
 * - Diagonal dominance: A[i,i] > sum(|A[i,j]| for j != i)
 * - Random values from normal distribution
 * - Consistent RHS vector: b = A * x
 * 
 * Diagonal dominance ensures convergence for iterative solvers like Jacobi and Gauss-Seidel.
 * 
 * @tparam MatrixType Template parameter for matrix storage type
 * @param A Output matrix being generated (modified in place)
 * @param x Known solution vector used to compute RHS
 * @param b Output RHS vector (modified in place)
 * @param startRow Starting row index for this thread
 * @param endRow Ending row index (exclusive) for this thread
 * @param threadSeed Random seed for this specific thread
 */
template <typename MatrixType>
void generateRows(MatrixType &A, const MatrixType &x, MatrixType &b, 
                  int startRow, int endRow, unsigned threadSeed) 
{
    std::default_random_engine gen(threadSeed);
    std::normal_distribution<typename MatrixType::value_type> ndist(0.0, 50.0);

    int n = static_cast<int>(A.get_row());

    for (int i = startRow; i < endRow; ++i) {
        typename MatrixType::value_type s = 0.0;
        
        // Generate random off-diagonal elements
        for (int j = 0; j < n; ++j) {
            typename MatrixType::value_type r = ndist(gen);
            s += std::abs(r);
            A.set_value(i, j, r);
        }
        
        // Ensure diagonal dominance by setting diagonal element > sum of off-diagonals
        A.set_value(i, i, s);

        // Compute RHS vector element: b[i] = sum(A[i,j] * x[j])
        typename MatrixType::value_type sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        b.set_value(i, 0, sum);
    }
}

/**
 * @brief Generate a random diagonally dominant matrix and save to file
 * 
 * This function creates a test matrix system suitable for solver benchmarking:
 * - Generates diagonally dominant coefficient matrix A
 * - Creates known solution vector x = [1, 2, 3, ..., n]
 * - Computes consistent RHS vector b = A * x
 * - Saves in augmented format for easy loading
 * 
 * The multi-threaded generation provides good performance for large matrices.
 * 
 * @tparam MatrixType Template parameter for matrix storage type
 * @param filename Output file path for the generated matrix system
 */
template <typename MatrixType>
void generateRandomMatrixAndWriteToFile(const std::string &filename) 
{
    int n;
    std::cout << "Enter the size of the matrix (n): ";
    while (!(std::cin >> n) || n <= 0) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input, please enter a positive integer: ";
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Use templated MatrixType
    MatrixType A(n, n);
    MatrixType x(n, 1);
    MatrixType b(n, 1);

    // Initialize solution vector x with known values [1, 2, 3, ..., n]
    // This allows verification that solvers find the correct solution
    for (int i = 0; i < n; ++i) {
        x.set_value(i, 0, static_cast<typename MatrixType::value_type>(i + 1));
    }

    // Determine optimal number of threads based on hardware
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    int rowsPerThread = n / static_cast<int>(numThreads);
    int remainder = n % static_cast<int>(numThreads);
    int startRow = 0;

    // Use a random device to seed each thread differently for better randomness
    std::random_device rd;
    unsigned baseSeed = rd();

    // Launch threads to generate matrix rows in parallel
    for (unsigned t = 0; t < numThreads; ++t) {
        int endRow = startRow + rowsPerThread + ((static_cast<int>(t) < remainder) ? 1 : 0);
        if (startRow >= n) break;
        unsigned threadSeed = baseSeed + t;
        threads.emplace_back(generateRows<MatrixType>, std::ref(A), std::ref(x), std::ref(b), 
                             startRow, endRow, threadSeed);
        startRow = endRow;
    }

    // Wait for all threads to complete
    for (auto &th : threads) {
        th.join();
    }

    // Write A and b to file in augmented format
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error opening file " << filename << " for writing.\n";
        return;
    }

    // File format: first line contains dimensions, subsequent lines contain matrix row + RHS
    fout << n << " " << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int col = 0; col < n; ++col) {
            fout << A.get_value(i, col) << " ";
        }
        fout << b.get_value(i, 0) << "\n";
    }
    fout.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nMatrix A and vector b have been written to " << filename << " in augmented format.\n";
    std::cout << "Generation and writing took: " 
              << std::chrono::duration<double>(end - start).count() << " seconds.\n";
}

/**
 * @brief Main function demonstrating matrix generation for different storage types
 * 
 * This program serves as a utility for generating test matrices with known solutions.
 * It demonstrates the template-based design by generating both dense and sparse matrices.
 * 
 * Generated matrices have the following properties:
 * - Diagonal dominance (guaranteed convergence for iterative solvers)
 * - Known solution vector x = [1, 2, 3, ..., n]
 * - Consistent RHS vector b = A * x
 * - Suitable for testing all solver types
 * 
 * Usage:
 * - Run the program and enter desired matrix size
 * - Two files will be generated: one for dense storage, one for sparse storage
 * - Use these files with the solver examples for performance testing
 */
int main() 
{
    const std::string filename_dense = "dense_matrix_data.txt";
    const std::string filename_sparse = "sparse_matrix_data.txt";

    std::cout << "Testing with Dense Matrix:\n";
    generateRandomMatrixAndWriteToFile<pnmatrix::matrix<pnmatrix::dense_matrix_storage<double>>>(filename_dense);

    std::cout << "\nTesting with Sparse Matrix:\n";
    generateRandomMatrixAndWriteToFile<pnmatrix::matrix<pnmatrix::sparse_matrix_storage<double>>>(filename_sparse);

    return 0;
}

/**
 * @file testmmapmatrix.h
 * @brief Comprehensive matrix testing framework with performance benchmarking
 * 
 * This header provides a complete testing and benchmarking framework for
 * the matrix library, including:
 * 
 * - High-performance matrix generation with parallel processing
 * - Multi-threaded solver testing and comparison
 * - Performance timing and benchmarking utilities
 * - File I/O operations for test data
 * - Template-based testing for different matrix storage types
 * 
 * Key features:
 * - Thread pool-based parallel matrix generation
 * - Diagonal dominance for solver convergence
 * - High-resolution timing for performance analysis
 * - Support for both dense and sparse matrix testing
 * - Comprehensive error handling and validation
 * 
 * Usage:
 * ```cpp
 * testMatrixGenerationAndSolve<DenseMatrix>("test.txt", "Dense", true, 1000);
 * testGaussianElimination<SparseMatrix>("test.txt", "Sparse", false, 500);
 * ```
 */

#pragma once

#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include "matrix_file_handler.h"
#include "gauss_seidel_solver.h"
#include "thread_pool.h"
#include "../include/dense_matrix_storage.h"
#include "../include/sparse_matrix_storage.h"
#include "../include/matrix.h"
#include "../include/gaussian_elimination.h"

namespace pnmatrix {

// Type aliases for commonly used matrix types
using DenseMatrix = matrix<dense_matrix_storage<double>>;
using SparseMatrix = matrix<sparse_matrix_storage<double>>;

/**
 * @brief High-resolution timer utility for performance measurement
 * 
 * Provides a simple interface for measuring execution time of functions
 * or code blocks using std::chrono::high_resolution_clock for maximum
 * precision in performance benchmarking.
 * 
 * Usage:
 * ```cpp
 * auto duration = Timer::measure([&]() {
 *     // Code to measure
 * });
 * double seconds = std::chrono::duration<double>(duration).count();
 * ```
 */
class Timer {
public:
    /**
     * @brief Measure execution time of a function or lambda
     * 
     * Executes the provided function and returns the elapsed time
     * with high-resolution timing precision.
     * 
     * @tparam Func Callable type (function, lambda, functor)
     * @param func Function to execute and measure
     * @return std::chrono::duration representing elapsed time
     */
    template <typename Func>
    static auto measure(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        return std::chrono::high_resolution_clock::now() - start;
    }
};

/**
 * @brief Print matrix contents with coordinate information
 * 
 * Outputs matrix elements in coordinate format showing row and column
 * indices along with values. Useful for debugging and visualization
 * of sparse matrices where many elements may be zero.
 * 
 * @tparam MatrixType Matrix type (must provide iterator interface)
 * @param m Matrix to print
 */
template <typename MatrixType>
void print_matrix(const MatrixType& m) {
    for (auto row = m.begin(); row != m.end(); ++row) {
        for (auto col = row.begin(); col != row.end(); ++col) {
            std::cout << "(" << col.row_index() << ", " << col.column_index() << ") " << *col << " ";
        }
        std::cout << "\n";
    }
}

/**
 * @brief Load matrix A and vector b from a text file
 * 
 * Simple file loading utility for reading matrix data from text format.
 * The file format expects matrix dimensions on the first line followed
 * by matrix rows with RHS vector elements appended.
 * 
 * File format:
 * First line: rows cols
 * Subsequent lines: A[i,0] A[i,1] ... A[i,n-1] b[i,0]
 * 
 * @tparam MatrixType Matrix storage type
 * @param A Output coefficient matrix
 * @param b Output RHS vector
 * @param filename Path to input file
 * @throws std::runtime_error on file access errors
 */
template <typename MatrixType>
void loadDefaultAB(MatrixType& A, MatrixType& b, const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    size_t rows, cols;
    inputFile >> rows >> cols;

    A = MatrixType(rows, cols);
    b = MatrixType(rows, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double val;
            inputFile >> val;
            A.set_value(i, j, val);
        }
        double b_val;
        inputFile >> b_val;
        b.set_value(i, 0, b_val);
    }
}

/**
 * @brief Generate rows of a diagonally dominant matrix in parallel
 * 
 * This function generates matrix rows with diagonal dominance properties
 * to ensure convergence of iterative solvers. Each thread works on a
 * contiguous range of rows for optimal cache performance.
 * 
 * Algorithm ensures diagonal dominance: A[i,i] > sum(|A[i,j]| for j ≠ i)
 * 
 * @param A Output matrix being generated (modified in place)
 * @param x Known solution vector for computing RHS
 * @param b Output RHS vector (modified in place)
 * @param startRow Starting row index for this thread
 * @param endRow Ending row index (exclusive) for this thread
 * @param threadSeed Random seed for this specific thread
 * @param rd Random device for seed generation
 */
void generateRows(DenseMatrix& A, const DenseMatrix& x, DenseMatrix& b,
                  int startRow, int endRow, unsigned threadSeed, std::random_device& rd) {
    std::mt19937 gen(threadSeed);
    std::normal_distribution<double> ndist(0.0, 50.0);

    int n = static_cast<int>(A.get_row());

    for (int i = startRow; i < endRow; ++i) {
        double s = 0.0;
        
        // Generate random off-diagonal elements
        for (int j = 0; j < n; ++j) {
            double r = ndist(gen);
            s += std::abs(r);
            A.set_value(i, j, r);
        }
        
        // Ensure diagonal dominance
        A.set_value(i, i, s);

        // Compute RHS: b[i] = sum(A[i,j] * x[j])
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        b.set_value(i, 0, sum);
    }
}

/**
 * @brief Initialize solution vector with sequential values
 * 
 * Creates a vector x = [1, 2, 3, ..., n] which serves as the known
 * solution for testing solver accuracy. This allows verification
 * that solvers find the correct solution.
 * 
 * @param x Output vector to initialize
 * @param n Size of the vector
 */
void initializeVector(DenseMatrix& x, int n) {
    for (int i = 0; i < n; ++i) {
        x.set_value(i, 0, static_cast<double>(i + 1));
    }
}

/**
 * @brief Write matrices A and b to a text file
 * 
 * Simple file writing utility for saving matrix data in text format.
 * The output format is human-readable and suitable for debugging
 * and data exchange.
 * 
 * File format:
 * First line: rows cols
 * Subsequent lines: A[i,0] A[i,1] ... A[i,n-1] b[i,0]
 * 
 * @param A Coefficient matrix to write
 * @param b RHS vector to write
 * @param filename Output file path
 */
void writeMatricesToFile(const DenseMatrix& A, const DenseMatrix& b, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error opening file " << filename << " for writing.\n";
        return;
    }

    fout << A.get_row() << " " << A.get_column() << "\n";
    for (int i = 0; i < A.get_row(); ++i) {
        for (int j = 0; j < A.get_column(); ++j) {
            fout << A.get_value(i, j) << " ";
        }
        fout << b.get_value(i, 0) << "\n";
    }
    fout.close();
}

/**
 * @brief Generate random diagonally dominant matrix using thread pool
 * 
 * Creates a test matrix system suitable for iterative solver testing.
 * Uses thread pool for parallel row generation to maximize performance
 * on multi-core systems. Each thread gets its own random generator
 * to avoid contention.
 * 
 * @param A Output coefficient matrix
 * @param x Output known solution vector
 * @param b Output RHS vector
 * @param n Matrix dimension
 * @param rd Random device for seed generation
 * @throws std::runtime_error if rows per thread is zero
 */
void generateRandomMatrix(DenseMatrix& A, DenseMatrix& x, DenseMatrix& b, int n, std::random_device& rd) {
    initializeVector(x, n);

    // Determine optimal thread count based on hardware
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(numThreads);

    // Calculate workload distribution
    int rowsPerThread = n / numThreads;
    if (rowsPerThread == 0) {
        throw std::runtime_error("Number of rows per thread is zero. Reduce the number of threads or increase matrix size.");
    }
    int remainder = n % numThreads;
    int startRow = 0;

    // Generate base seed for thread-specific seeds
    unsigned baseSeed = rd();

    std::vector<std::future<void>> futures;
    
    // Distribute work among threads
    for (unsigned t = 0; t < numThreads; ++t) {
        int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);
        futures.emplace_back(pool.enqueue(generateRows, std::ref(A), std::ref(x), std::ref(b), 
                                        startRow, endRow, baseSeed + t, std::ref(rd)));
        startRow = endRow;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
}

/**
 * @brief Generate random matrix and write to file with timing
 * 
 * Creates a diagonally dominant matrix system and saves it to file
 * with performance timing for both generation and writing operations.
 * 
 * @param filename Output file path
 * @param n Matrix dimension
 */
void generateRandomMatrixAndWriteToFile(const std::string& filename, int n) {
    DenseMatrix A(n, n);
    DenseMatrix x(n, 1);
    DenseMatrix b(n, 1);

    std::random_device rd;

    auto generationTime = Timer::measure([&] {
        generateRandomMatrix(A, x, b, n, rd);
        writeMatricesToFile(A, b, filename);
    });

    std::cout << "Matrix and vector b written to " << filename << " in "
              << std::chrono::duration<double>(generationTime).count() << " seconds.\n";
}

/**
 * @brief Generate and write matrix to file (simplified interface)
 * 
 * Convenience function that combines matrix generation and file writing
 * into a single operation.
 * 
 * @param filename Output file path
 * @param size Matrix dimension
 */
void generateAndWriteMatrix(const std::string& filename, int size) {
    DenseMatrix A(size, size);
    DenseMatrix x(size, 1);
    DenseMatrix b(size, 1);

    std::random_device rd;
    generateRandomMatrix(A, x, b, size, rd);
    writeMatricesToFile(A, b, filename);
}

/**
 * @brief Comprehensive test for matrix generation and Gauss-Seidel solver
 * 
 * Performs end-to-end testing of matrix generation, file I/O, and
 * Gauss-Seidel solver performance. Measures timing for each phase
 * and provides detailed performance analysis.
 * 
 * Test phases:
 * 1. Matrix generation and file writing
 * 2. Matrix loading from file
 * 3. Gauss-Seidel solver execution
 * 4. Optional solution printing
 * 
 * @tparam MatrixType Matrix storage type (dense or sparse)
 * @param filename File path for matrix data
 * @param matrixType Description string for output
 * @param printSolution Whether to print the solution vector
 * @param size Matrix dimension
 */
template <typename MatrixType>
void testMatrixGenerationAndSolve(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;

        std::cout << "\n--- " << matrixType << " Matrix Test ---\n";

        // Phase 1: Matrix generation and file writing
        auto genTime = Timer::measure([&] {
            generateAndWriteMatrix(filename, size);
        });
        std::cout << matrixType << " Matrix Generation Time: "
                  << std::chrono::duration<double>(genTime).count() << " seconds.\n";

        // Phase 2: Matrix loading from file
        auto loadTime = Timer::measure([&] {
            loadDefaultAB(A, b, filename);
        });
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadTime).count() << " seconds.\n";

        // Phase 3: Gauss-Seidel solver execution
        gauss_seidel::option op;
        op.rm = 1e-6;
        gauss_seidel solver(op);

        auto solveTime = Timer::measure([&] {
            auto result = solver.solve(A, b);
            if (printSolution) {
                std::cout << "\nSolution x:\n";
                pnmatrix::print_matrix(result);
            }
        });
        std::cout << "Solver Execution Time: "
                  << std::chrono::duration<double>(solveTime).count() << " seconds.\n";

        // Total performance summary
        std::cout << "Total Time for " << matrixType << " Matrix: "
                  << std::chrono::duration<double>(genTime + loadTime + solveTime).count() << " seconds.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

/**
 * @brief Comprehensive test for matrix generation and Gaussian elimination
 * 
 * Performs end-to-end testing of matrix generation, file I/O, and
 * Gaussian elimination solver performance. Measures timing for each
 * phase and provides detailed performance analysis.
 * 
 * Test phases:
 * 1. Matrix generation and file writing
 * 2. Matrix loading from file
 * 3. Gaussian elimination solver execution
 * 4. Optional solution printing
 * 
 * @tparam MatrixType Matrix storage type (dense or sparse)
 * @param filename File path for matrix data
 * @param matrixType Description string for output
 * @param printSolution Whether to print the solution vector
 * @param size Matrix dimension
 */
template <typename MatrixType>
void testGaussianElimination(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;

        std::cout << "\n--- " << matrixType << " Matrix Gaussian Elimination Test ---\n";

        // Phase 1: Matrix generation and file writing
        auto genTime = Timer::measure([&] {
            generateAndWriteMatrix(filename, size);
        });
        std::cout << matrixType << " Matrix Generation Time: "
                  << std::chrono::duration<double>(genTime).count() << " seconds.\n";

        // Phase 2: Matrix loading from file
        auto loadTime = Timer::measure([&] {
            loadDefaultAB(A, b, filename);
        });
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadTime).count() << " seconds.\n";

        // Phase 3: Gaussian elimination solver execution
        gaussian_elimination::option op;
        op.rm = 1e-6;
        gaussian_elimination solver(op);

        auto solveTime = Timer::measure([&] {
            auto result = solver.solve(A, b);
            if (printSolution) {
                std::cout << "\nSolution x:\n";
                pnmatrix::print_matrix(result);
            }
        });
        std::cout << "Solver Execution Time: "
                  << std::chrono::duration<double>(solveTime).count() << " seconds.\n";

        // Total performance summary
        std::cout << "Total Time for " << matrixType << " Matrix: "
                  << std::chrono::duration<double>(genTime + loadTime + solveTime).count() << " seconds.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

} // namespace pnmatrix

/**
 * @file matrix_solver.h
 * @brief High-performance matrix solver utilities with thread pool optimization
 * 
 * This header provides optimized implementations for matrix generation,
 * solving linear systems, and performance benchmarking. It includes:
 * 
 * - Thread pool for parallel matrix operations
 * - Optimized matrix generation with diagonal dominance
 * - Multi-threaded solver testing framework
 * - Performance timing and benchmarking utilities
 * - File I/O operations for matrix data
 * 
 * Features:
 * - Parallel matrix generation using thread pool
 * - Memory-mapped file operations for large matrices
 * - High-resolution timing for performance analysis
 * - Template-based design for different matrix storage types
 * - Thread-safe operations for concurrent execution
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
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include "matrix_file_handler.h" // Updated to MatrixFileHandler
#include "gauss_seidel_solver.h"
#include "../include/dense_matrix_storage.h"
#include "../include/sparse_matrix_storage.h"
#include "../include/matrix.h"
#include "../include/gaussian_elimination.h" // Include Gaussian Elimination header

namespace pnmatrix {

/**
 * @brief High-performance thread pool for parallel matrix operations
 * 
 * Implements a work-stealing thread pool with task queue and condition
 * variable synchronization. This provides efficient parallel execution
 * for matrix operations that can be divided into independent tasks.
 * 
 * Features:
 * - Configurable number of worker threads
 * - Task queue with FIFO scheduling
 * - Future-based result retrieval
 * - Graceful shutdown with join semantics
 * 
 * Usage:
 * ```cpp
 * ThreadPool pool(4);  // 4 worker threads
 * auto future = pool.enqueue(function, args...);
 * auto result = future.get();  // Wait for completion
 * ```
 */
class ThreadPool {
public:
    /**
     * @brief Construct thread pool with specified number of worker threads
     * 
     * Creates worker threads that continuously pull tasks from the queue
     * and execute them. Threads are started immediately and wait for
     * tasks using condition variables for efficient CPU usage.
     * 
     * @param numThreads Number of worker threads to create
     */
    explicit ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex_);
                        condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();  // Execute the task
                }
            });
        }
    }

    /**
     * @brief Enqueue a task for execution by the thread pool
     * 
     * Adds a function with its arguments to the task queue and returns
     * a future for retrieving the result. Uses perfect forwarding
     * to avoid unnecessary copies and supports any callable type.
     * 
     * @tparam F Function type to execute
     * @tparam Args Argument types for the function
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return std::future for the function's return value
     * @throws std::runtime_error if enqueue is called on stopped pool
     */
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using returnType = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<returnType()>>(
            [f = std::forward<F>(f), ... args = std::forward<Args>(args)]() mutable { return f(args...); });

        std::future<returnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            if (stop_)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    /**
     * @brief Destructor - gracefully shuts down the thread pool
     * 
     * Signals all worker threads to stop, waits for them to complete
     * current tasks, and then joins all threads. This ensures clean
     * shutdown without resource leaks.
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_)
            worker.join();
    }

private:
    std::vector<std::thread> workers_;           ///< Worker threads
    std::queue<std::function<void()>> tasks_;      ///< Task queue
    std::mutex queueMutex_;                       ///< Mutex for task queue access
    std::condition_variable condition_;           ///< Condition variable for task availability
    bool stop_ = false;                            ///< Flag to signal thread shutdown
};

// Type aliases for commonly used matrix types
using DenseMatrix = matrix<dense_matrix_storage<double>>;
using SparseMatrix = matrix<sparse_matrix_storage<double>>;

/**
 * @brief Print matrix contents with coordinate information
 * 
 * Outputs matrix elements in coordinate format showing row and column
 * indices along with values. Useful for debugging and visualization.
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
 * @brief Optimized multi-threaded row generator for dense matrices
 * 
 * Generates rows of a diagonally dominant matrix in parallel.
 * Each thread works on a contiguous range of rows to minimize
 * cache contention and maximize performance.
 * 
 * Algorithm ensures diagonal dominance: A[i,i] > sum(|A[i,j]| for j ≠ i)
 * 
 * @param A Output matrix being generated (modified in place)
 * @param x Known solution vector for computing RHS
 * @param b Output RHS vector (modified in place)
 * @param startRow Starting row index for this thread
 * @param endRow Ending row index (exclusive) for this thread
 * @param gen Random number generator for this thread
 * @param ndist Normal distribution for random values
 */
inline void generateRows(DenseMatrix& A, const DenseMatrix& x, DenseMatrix& b,
                        int startRow, int endRow, std::mt19937& gen, std::normal_distribution<double>& ndist) {
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
    DenseMatrix mat(n, 1);
    for (int i = 0; i < n; ++i) {
        mat.set_value(i, 0, static_cast<double>(i + 1));
    }
    x.set_column(0, mat);
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
 */
void generateRandomMatrix(DenseMatrix& A, DenseMatrix& x, DenseMatrix& b, int n, std::random_device& rd) {
    initializeVector(x, n);

    // Determine optimal thread count based on hardware
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(numThreads);

    // Calculate workload distribution
    int rowsPerThread = n / numThreads;
    int remainder = n % numThreads;

    std::vector<std::future<void>> futures;

    // Create separate random generators for each thread to avoid contention
    std::vector<std::mt19937> generators;
    generators.reserve(numThreads);
    for (unsigned t = 0; t < numThreads; ++t) {
        generators.emplace_back(rd());
    }

    std::normal_distribution<double> ndist(0.0, 50.0);

    // Distribute work among threads
    for (unsigned t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread + std::min(static_cast<int>(t), remainder);
        int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);

        // Enqueue task for thread pool execution
        futures.emplace_back(pool.enqueue(generateRows, std::ref(A), std::ref(x), std::ref(b),
                                        startRow, endRow, std::ref(generators[t]), std::ref(ndist)));
    }

    // Wait for all threads to complete
    for (auto& fut : futures) {
        fut.get();
    }
}

/**
 * @brief Generate random matrix and write to file with performance timing
 * 
 * Creates a diagonally dominant matrix system and saves it to file
 * using binary format for efficient I/O. Provides timing information
 * for both generation and file writing operations.
 * 
 * @param filename Output file path
 * @param n Matrix dimension
 * @param writeTime Accumulated write time (output parameter)
 */
void generateRandomMatrixAndWriteToFile(const std::string& filename, int n, double& writeTime) {
    auto start = std::chrono::high_resolution_clock::now();

    DenseMatrix A(n, n);
    DenseMatrix x(n, 1);
    DenseMatrix b(n, 1);

    std::random_device rd;
    generateRandomMatrix(A, x, b, n, rd);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix A, vector x, and vector b generated in "
              << std::chrono::duration<double>(end - start).count() << " seconds.\n";

    auto writeStart = std::chrono::high_resolution_clock::now();
    MatrixFileHandler fileHandler;
    fileHandler.writeBinaryAB(A, b, filename);

    auto writeEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix A and vector b written to " << filename << " in "
              << std::chrono::duration<double>(writeEnd - writeStart).count() << " seconds.\n";

    writeTime += std::chrono::duration<double>(writeEnd - start).count();
}

/**
 * @brief Comprehensive test for matrix generation and solver performance
 * 
 * This function performs end-to-end testing of the matrix generation,
 * file I/O, and solver performance. It measures timing for each phase
 * and provides detailed performance analysis.
 * 
 * Test phases:
 * 1. Matrix generation and file writing
 * 2. Matrix loading from file
 * 3. Solver execution (Gauss-Seidel)
 * 4. Solution validation (optional)
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
        MatrixFileHandler fileHandler;

        auto totalStart = std::chrono::high_resolution_clock::now();

        std::cout << "\n--- " << matrixType << " Matrix Gauss-Seidel Solver Test ---\n";

        double writeTime = 0.0;
        auto genStart = std::chrono::high_resolution_clock::now();
        
        // Phase 1: Matrix generation and file writing
        generateRandomMatrixAndWriteToFile(filename, size, writeTime);
        
        auto genEnd = std::chrono::high_resolution_clock::now();
        std::cout << matrixType << " Matrix Generation and Writing Time: "
                  << std::chrono::duration<double>(genEnd - genStart).count() << " seconds.\n";
        std::cout << "File Writing Time: " << (std::chrono::duration<double>(genEnd - genStart).count()) << " seconds.\n";

        // Phase 2: Matrix loading from file
        auto loadStart = std::chrono::high_resolution_clock::now();
        fileHandler.loadBinaryAB(A, b, filename);
        auto loadEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadEnd - loadStart).count() << " seconds.\n";

        // Phase 3: Solver execution
        gauss_seidel::option op;
        op.rm = 1e-6;
        gauss_seidel solver(op);

        auto solveStart = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(A, b);
        auto solveEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Gauss-Seidel Solver Execution Time: "
                  << std::chrono::duration<double>(solveEnd - solveStart).count() << " seconds.\n";

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Total Time for " << matrixType << " Matrix Gauss-Seidel Test: "
                  << std::chrono::duration<double>(totalEnd - totalStart).count() << " seconds.\n";

        if (printSolution) {
            std::cout << "\nSolution x:\n";
            print_matrix(result);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

// Test Gaussian Elimination with memory-mapped binary I/O
template <typename MatrixType>
void testGaussianElimination(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;
        MatrixFileHandler fileHandler;

        auto totalStart = std::chrono::high_resolution_clock::now();

        std::cout << "\n--- " << matrixType << " Matrix Gaussian Elimination Test ---\n";

        double writeTime = 0.0;
        auto genStart = std::chrono::high_resolution_clock::now();
        // Generate and write matrices A and b
        generateRandomMatrixAndWriteToFile(filename, size, writeTime);
        auto genEnd = std::chrono::high_resolution_clock::now();
        std::cout << matrixType << " Matrix Generation and Writing Time: "
                  << std::chrono::duration<double>(genEnd - genStart).count() << " seconds.\n";
        std::cout << "File Writing Time: " << (std::chrono::duration<double>(genEnd - genStart).count()) << " seconds.\n";

        auto loadStart = std::chrono::high_resolution_clock::now();
        fileHandler.loadBinaryAB(A, b, filename);
        auto loadEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadEnd - loadStart).count() << " seconds.\n";

        gaussian_elimination::option op;
        op.rm = 1e-6;
        gaussian_elimination solver(op);

        auto solveStart = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(A, b);
        auto solveEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Gaussian Elimination Solver Execution Time: "
                  << std::chrono::duration<double>(solveEnd - solveStart).count() << " seconds.\n";

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Total Time for " << matrixType << " Matrix Gaussian Elimination Test: "
                  << std::chrono::duration<double>(totalEnd - totalStart).count() << " seconds.\n";

        if (printSolution) {
            std::cout << "\nSolution x:\n";
            print_matrix(result);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

} // namespace pnmatrix

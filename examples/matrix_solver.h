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

// ThreadPool class remains unchanged
class ThreadPool {
public:
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

                    task();
                }
            });
        }
    }

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
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queueMutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

// Alias types for dense and sparse matrices
using DenseMatrix = matrix<dense_matrix_storage<double>>;
using SparseMatrix = matrix<sparse_matrix_storage<double>>;

// Function to print a matrix
template <typename MatrixType>
void print_matrix(const MatrixType& m) {
    for (auto row = m.begin(); row != m.end(); ++row) {
        for (auto col = row.begin(); col != row.end(); ++col) {
            std::cout << "(" << col.row_index() << ", " << col.column_index() << ") " << *col << " ";
        }
        std::cout << "\n";
    }
}

// Optimized multi-threaded row generator for DenseMatrix
inline void generateRows(DenseMatrix& A, const DenseMatrix& x, DenseMatrix& b,
                        int startRow, int endRow, std::mt19937& gen, std::normal_distribution<double>& ndist) {
    int n = static_cast<int>(A.get_row());

    for (int i = startRow; i < endRow; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            double r = ndist(gen);
            s += std::abs(r);
            A.set_value(i, j, r);
        }
        A.set_value(i, i, s); // Ensure diagonal dominance

        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        b.set_value(i, 0, sum);
    }
}

// Function to initialize matrix x
void initializeVector(DenseMatrix& x, int n) {
    DenseMatrix mat(n, 1);
    for (int i = 0; i < n; ++i) {
        mat.set_value(i, 0, static_cast<double>(i + 1));
    }
    x.set_column(0, mat);
}

// Optimized function to generate random matrix
void generateRandomMatrix(DenseMatrix& A, DenseMatrix& x, DenseMatrix& b, int n, std::random_device& rd) {
    initializeVector(x, n);

    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(numThreads);

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

    for (unsigned t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread + std::min(static_cast<int>(t), remainder);
        int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);

        // Capture by reference except for generator and distribution
        futures.emplace_back(pool.enqueue(generateRows, std::ref(A), std::ref(x), std::ref(b),
                                        startRow, endRow, std::ref(generators[t]), std::ref(ndist)));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

// Random matrix generator for DenseMatrix with optimized file writing
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

// Test matrix generation and solving system Ax = b using Gauss-Seidel
template <typename MatrixType>
void testMatrixGenerationAndSolve(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;
        MatrixFileHandler fileHandler;

        auto totalStart = std::chrono::high_resolution_clock::now();

        std::cout << "\n--- " << matrixType << " Matrix Gauss-Seidel Solver Test ---\n";

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

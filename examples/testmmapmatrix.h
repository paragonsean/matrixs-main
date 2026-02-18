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

using DenseMatrix = matrix<dense_matrix_storage<double>>;
using SparseMatrix = matrix<sparse_matrix_storage<double>>;

class Timer {
public:
    template <typename Func>
    static auto measure(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        return std::chrono::high_resolution_clock::now() - start;
    }
};
template <typename MatrixType>
void print_matrix(const MatrixType& m) {
    for (auto row = m.begin(); row != m.end(); ++row) {
        for (auto col = row.begin(); col != row.end(); ++col) {
            std::cout << "(" << col.row_index() << ", " << col.column_index() << ") " << *col << " ";
        }
        std::cout << "\n";
    }
}
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

void generateRows(DenseMatrix& A, const DenseMatrix& x, DenseMatrix& b,
                  int startRow, int endRow, unsigned threadSeed, std::random_device& rd) {
    std::mt19937 gen(threadSeed);
    std::normal_distribution<double> ndist(0.0, 50.0);

    int n = static_cast<int>(A.get_row());

    for (int i = startRow; i < endRow; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            double r = ndist(gen);
            s += std::abs(r);
            A.set_value(i, j, r);
        }
        A.set_value(i, i, s);

        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        b.set_value(i, 0, sum);
    }
}

void initializeVector(DenseMatrix& x, int n) {
    for (int i = 0; i < n; ++i) {
        x.set_value(i, 0, static_cast<double>(i + 1));
    }
}

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

void generateRandomMatrix(DenseMatrix& A, DenseMatrix& x, DenseMatrix& b, int n, std::random_device& rd) {
    initializeVector(x, n);

    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(numThreads);

    int rowsPerThread = n / numThreads;
    if (rowsPerThread == 0) {
        throw std::runtime_error("Number of rows per thread is zero. Reduce the number of threads or increase matrix size.");
    }
    int remainder = n % numThreads;
    int startRow = 0;

    unsigned baseSeed = rd();

    std::vector<std::future<void>> futures;
    for (unsigned t = 0; t < numThreads; ++t) {
        int endRow = startRow + rowsPerThread + (t < remainder ? 1 : 0);
        futures.emplace_back(pool.enqueue(generateRows, std::ref(A), std::ref(x), std::ref(b), startRow, endRow, baseSeed + t, std::ref(rd)));
        startRow = endRow;
    }

    for (auto& future : futures) {
        future.get();
    }
}

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

void generateAndWriteMatrix(const std::string& filename, int size) {
    DenseMatrix A(size, size);
    DenseMatrix x(size, 1);
    DenseMatrix b(size, 1);

    std::random_device rd;
    generateRandomMatrix(A, x, b, size, rd);
    writeMatricesToFile(A, b, filename);
}

template <typename MatrixType>
void testMatrixGenerationAndSolve(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;

        std::cout << "\n--- " << matrixType << " Matrix Test ---\n";

        auto genTime = Timer::measure([&] {
            generateAndWriteMatrix(filename, size);
        });
        std::cout << matrixType << " Matrix Generation Time: "
                  << std::chrono::duration<double>(genTime).count() << " seconds.\n";

        auto loadTime = Timer::measure([&] {
            loadDefaultAB(A, b, filename);
        });
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadTime).count() << " seconds.\n";

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

        std::cout << "Total Time for " << matrixType << " Matrix: "
                  << std::chrono::duration<double>(genTime + loadTime + solveTime).count() << " seconds.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

template <typename MatrixType>
void testGaussianElimination(const std::string& filename, const std::string& matrixType, bool printSolution, int size) {
    try {
        MatrixType A, b;

        std::cout << "\n--- " << matrixType << " Matrix Gaussian Elimination Test ---\n";

        auto genTime = Timer::measure([&] {
            generateAndWriteMatrix(filename, size);
        });
        std::cout << matrixType << " Matrix Generation Time: "
                  << std::chrono::duration<double>(genTime).count() << " seconds.\n";

        auto loadTime = Timer::measure([&] {
            loadDefaultAB(A, b, filename);
        });
        std::cout << "Matrix Load Time: "
                  << std::chrono::duration<double>(loadTime).count() << " seconds.\n";

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

        std::cout << "Total Time for " << matrixType << " Matrix: "
                  << std::chrono::duration<double>(genTime + loadTime + solveTime).count() << " seconds.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

} // namespace pnmatrix

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


// Templated function to load Matrix A and vector b from a file
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

template <typename MatrixType>
void generateRows(MatrixType &A, const MatrixType &x, MatrixType &b, 
                  int startRow, int endRow, unsigned threadSeed) 
{
    std::default_random_engine gen(threadSeed);
    std::normal_distribution<typename MatrixType::value_type> ndist(0.0, 50.0);

    int n = static_cast<int>(A.get_row());

    for (int i = startRow; i < endRow; ++i) {
        typename MatrixType::value_type s = 0.0;
        for (int j = 0; j < n; ++j) {
            typename MatrixType::value_type r = ndist(gen);
            s += std::abs(r);
            A.set_value(i, j, r);
        }
        // Ensure diagonal dominance
        A.set_value(i, i, s);

        // Compute b[i,0]
        typename MatrixType::value_type sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        b.set_value(i, 0, sum);
    }
}

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

    // Initialize x with values from 1 to n
    for (int i = 0; i < n; ++i) {
        x.set_value(i, 0, static_cast<typename MatrixType::value_type>(i + 1));
    }

    // Determine number of threads
    unsigned numThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    int rowsPerThread = n / static_cast<int>(numThreads);
    int remainder = n % static_cast<int>(numThreads);
    int startRow = 0;

    // Use a random device to seed each thread differently
    std::random_device rd;
    unsigned baseSeed = rd();

    for (unsigned t = 0; t < numThreads; ++t) {
        int endRow = startRow + rowsPerThread + ((static_cast<int>(t) < remainder) ? 1 : 0);
        if (startRow >= n) break;
        unsigned threadSeed = baseSeed + t;
        threads.emplace_back(generateRows<MatrixType>, std::ref(A), std::ref(x), std::ref(b), 
                             startRow, endRow, threadSeed);
        startRow = endRow;
    }

    for (auto &th : threads) {
        th.join();
    }

    // Write A and b to file in augmented format
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error opening file " << filename << " for writing.\n";
        return;
    }

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

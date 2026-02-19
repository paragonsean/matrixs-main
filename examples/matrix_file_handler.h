/**
 * @file matrix_file_handler.h
 * @brief High-performance matrix file I/O with memory mapping and parallel processing
 * 
 * This header provides efficient file I/O operations for matrix data, including
 * both text and binary formats with advanced optimizations:
 * 
 * Features:
 * - Memory-mapped file operations for large matrices
 * - Parallel file reading/writing with multi-threading
 * - Template-based design for different matrix storage types
 * - High-resolution timing for performance analysis
 * - Memory-efficient string construction
 * - Exception-safe error handling
 * 
 * Supported formats:
 * - Text format: Human-readable augmented matrix format
 * - Binary format: Memory-mapped binary for maximum I/O performance
 * 
 * Performance optimizations:
 * - Memory mapping for large file operations
 * - Parallel processing for multi-core systems
 * - Efficient string construction with pre-allocation
 * - Asynchronous I/O with std::async
 * - Hardware-aware thread allocation
 */

#ifndef MATRIX_FILE_HANDLER_H
#define MATRIX_FILE_HANDLER_H

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <random>
#include <ctime>
#include <thread>
#include <vector>
#include <atomic>
#include <sstream>
#include <future> // For std::async and std::future
#include <algorithm> // For std::min and std::max
#include <sys/mman.h> // For mmap
#include <sys/stat.h>  // For fstat
#include <fcntl.h>     // For open
#include <unistd.h>    // For close, ftruncate
#include "../include/matrix.h"
#include "../include/dense_matrix_storage.h"
#include "../include/sparse_matrix_storage.h"

namespace pnmatrix {

/**
 * @brief High-performance matrix file I/O handler
 * 
 * This class provides efficient file operations for matrix data with support
 * for both dense and sparse matrix storage types. It implements advanced
 * techniques for high-performance I/O including memory mapping and parallel
 * processing.
 * 
 * Key capabilities:
 * - Binary file I/O with memory mapping for large matrices
 * - Text file I/O with parallel processing
 * - Template-based design for storage abstraction
 * - Performance timing and benchmarking
 * - Exception-safe error handling
 * 
 * Design philosophy:
 * - Memory efficiency: Use memory mapping for large files
 * - CPU efficiency: Parallel processing for multi-core systems
 * - Flexibility: Template-based for different storage types
 * - Robustness: Comprehensive error checking and handling
 */
class MatrixFileHandler {
public:
    /**
     * @brief Load matrices A and b from a binary file using memory mapping
     * 
     * Implements high-performance binary file reading using memory mapping
     * for optimal I/O performance, especially for large matrices. The binary
     * format stores matrix data in native binary representation for
     * maximum efficiency.
     * 
     * Binary format:
     * - Header: rows, cols (as size_t)
     - Matrix A data: rows * cols elements (as value_type)
     * Vector b data: rows elements (as value_type)
     * 
     * @tparam MatrixType Matrix storage type (dense or sparse)
     * @param A Output coefficient matrix
     * @param b Output RHS vector
     * @param filename Path to binary file
     * @throws std::runtime_error on file access errors
     */
    template <typename MatrixType>
    void loadBinaryAB(MatrixType& A, MatrixType& b, const std::string& filename);

    /**
     * @brief Write matrices A and b to a binary file using memory mapping
     * 
     * Stores matrix data in binary format using memory mapping for
     * efficient I/O. This method is optimized for large matrices where
     * traditional file I/O would be a bottleneck.
     * 
     * @tparam MatrixType Matrix storage type (dense or sparse)
     * @param A Coefficient matrix to write
     * @param b RHS vector to write
     * @param filename Output file path
     * @throws std::runtime_error on file access errors
     */
    template <typename MatrixType>
    void writeBinaryAB(const MatrixType& A, const MatrixType& b, const std::string& filename);
    
    /**
     * @brief Load matrices A and b from a text file with parallel processing
     * 
     * Reads matrix data from a human-readable text format with parallel
     * processing for improved performance on multi-core systems. The text
     * format stores matrix rows with the RHS vector element appended.
     * 
     * Text format:
     * First line: rows cols
     * Subsequent lines: A[i,0] A[i,1] ... A[i,n-1] b[i,0]
     * 
     * @tparam MatrixType Matrix storage type (dense or sparse)
     * @param A Output coefficient matrix
     * @param b Output RHS vector
     * @param filename Path to text file
     * @throws std::runtime_error on file access errors
     */
    template <typename MatrixType>
    void loadDefaultAB(MatrixType& A, MatrixType& b, const std::string& filename);

    /**
     * @brief Write matrices A and b to a text file with parallel processing
     * 
     * Writes matrix data in human-readable text format using asynchronous
     * I/O and parallel string construction for optimal performance.
     * This method is suitable for debugging and data exchange.
     * 
     * @tparam MatrixType Matrix storage type (dense or sparse)
     * @param A Coefficient matrix to write
     * @param b RHS vector to write
     * @param filename Output file path
     * @throws std::runtime_error on file access errors
     */
    template <typename MatrixType>
    void writeDefaultAB(const MatrixType& A, const MatrixType& b, const std::string& filename);

private:
    /**
     * @brief Parallel parsing of text file lines for matrix data
     * 
     * Implements parallel processing of matrix rows from text format.
     * Each thread processes a contiguous chunk of rows to minimize
     * cache contention and maximize performance.
     * 
     * @tparam MatrixType Matrix storage type
     * @param lines Vector of text lines to parse
     * @param A Output coefficient matrix
     * @param b Output RHS vector
     */
    template <typename MatrixType>
    void parallelParseLines(std::vector<std::string>& lines, MatrixType& A, MatrixType& b);
};

/**
 * @brief Implementation of parallel line parsing for text matrix files
 * 
 * This function divides the parsing work among multiple threads to
 * maximize performance on multi-core systems. Each thread processes
 * a contiguous range of matrix rows, parsing the text format and
 * populating the matrix objects.
 * 
 * The parsing algorithm expects each line to contain matrix elements
 * followed by the RHS vector element, separated by whitespace.
 * 
 * @tparam MatrixType Matrix storage type
 * @param lines Vector of text lines to parse
 * @param A Output coefficient matrix (modified in place)
 * @param b Output RHS vector (modified in place)
 */
template <typename MatrixType>
void MatrixFileHandler::parallelParseLines(std::vector<std::string>& lines, MatrixType& A, MatrixType& b) {
    // Lambda function for processing a chunk of rows
    auto parseChunk = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            std::istringstream iss(lines[i]);
            
            // Parse matrix elements from the line
            for (size_t j = 0; j < A.get_column(); ++j) {
                double val;
                iss >> val;
                A.set_value(i, j, val);
            }
            
            // Parse RHS vector element from the line
            double b_val;
            iss >> b_val;
            b.set_value(i, 0, b_val);
        }
    };

    // Determine optimal thread count based on data size and hardware
    unsigned numThreads = std::min(std::thread::hardware_concurrency(), static_cast<unsigned>(lines.size()));
    if (numThreads == 0) numThreads = 1; // Fallback to 1 thread if hardware_concurrency returns 0
    
    std::vector<std::thread> threads;
    size_t chunkSize = (lines.size() + numThreads - 1) / numThreads;

    // Launch threads for parallel processing
    for (unsigned t = 0; t < numThreads; ++t) {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, lines.size());
        if (start >= end) break; // Avoid creating threads with no work
        threads.emplace_back(parseChunk, start, end);
    }

    // Wait for all threads to complete
    for (auto& th : threads) th.join();
}

/**
 * @brief Implementation of text file matrix loading with parallel processing
 * 
 * Reads matrix data from text format, validates dimensions, and uses
 * parallel processing for optimal performance. Provides detailed
 * timing information and comprehensive error checking.
 */
template <typename MatrixType>
void MatrixFileHandler::loadDefaultAB(MatrixType& A, MatrixType& b, const std::string& filename) {
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        throw std::runtime_error("Failed to open file for reading.");
    }

    // Read matrix dimensions from header
    size_t rows, cols;
    if (!(inputFile >> rows >> cols)) {
        std::cerr << "Failed to read dimensions for matrix A from file.\n";
        throw std::runtime_error("Failed to read matrix dimensions.");
    }
    if (rows == 0 || cols == 0) {
        std::cerr << "Matrix A dimensions must be greater than zero.\n";
        throw std::runtime_error("Invalid matrix dimensions.");
    }

    // Initialize matrices with correct dimensions
    A = MatrixType(rows, cols);
    b = MatrixType(rows, 1);
    
    // Read all lines into memory for parallel processing
    std::vector<std::string> lines(rows);
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore the rest of the header line
    for (size_t i = 0; i < rows; ++i) {
        if (!std::getline(inputFile, lines[i])) {
            std::cerr << "Error reading line " << i << " from file.\n";
            throw std::runtime_error("Error reading matrix data.");
        }
    }
    inputFile.close();
    
    // Parse lines in parallel
    parallelParseLines(lines, A, b);

    // Debug: Print loaded dimensions and sample values
    std::cout << "Loaded Matrix A: " << A.get_row() << "x" << A.get_column() << "\n";
    std::cout << "Loaded Vector b: " << b.get_row() << "x" << b.get_column() << "\n";

    // Validate matrix dimensions after loading
    if (A.get_column() != b.get_row() || b.get_column() != 1) {
        std::cerr << "Dimension mismatch after loading: A(" << A.get_row() << "x" << A.get_column()
                  << "), b(" << b.get_row() << "x" << b.get_column() << ")\n";
        throw std::invalid_argument("Dimension mismatch between A and b.");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix A and vector b successfully loaded from " << filename << "\n";
    std::cout << "Loading took: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";
}

/**
 * @brief Implementation of parallel text file writing with asynchronous I/O
 * 
 * Writes matrix data to text format using asynchronous I/O and parallel
 * string construction. This method optimizes performance by:
 * 
 * - Using std::async for non-blocking I/O operations
 * - Pre-allocating string buffers to avoid reallocations
 * - Parallel string construction for CPU-intensive operations
 * - Efficient memory management with move semantics
 * 
 * The output format is human-readable and suitable for debugging
 * and data exchange between different systems.
 */
template <typename MatrixType>
void MatrixFileHandler::writeDefaultAB(const MatrixType& A, const MatrixType& b, const std::string& filename) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t rows = A.get_row();
    size_t cols = A.get_column();

    // Validate matrix dimensions
    if (b.get_row() != rows || b.get_column() != 1) {
        std::cerr << "Dimension mismatch: Matrix A is " << rows << "x" << cols
                  << ", but vector b is " << b.get_row() << "x" << b.get_column() << "\n";
        throw std::invalid_argument("Dimension mismatch between A and b.");
    }

    // Open output file
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error opening file " << filename << " for writing.\n";
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Write header with matrix dimensions
    fout << rows << " " << cols << "\n";

    // Set fixed precision for consistent output format
    fout << std::fixed << std::setprecision(6);

    // Determine optimal thread count based on data size and hardware
    unsigned numThreads = std::min(static_cast<unsigned>(std::thread::hardware_concurrency()), static_cast<unsigned>(rows));
    if (numThreads == 0) numThreads = 1; // Fallback to 1 thread if hardware_concurrency returns 0

    size_t chunkSize = (rows + numThreads - 1) / numThreads;
    std::vector<std::future<std::vector<std::string>>> futures;

    // Launch asynchronous tasks for parallel string construction
    for (unsigned t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, rows);

        if (startIdx >= endIdx) break; // No work for this thread

        // Async task for constructing row strings
        futures.emplace_back(std::async(std::launch::async, [&](size_t start, size_t end) -> std::vector<std::string> {
            std::vector<std::string> localRows;
            localRows.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                std::string rowStr;
                rowStr.reserve(cols * 20 + 10); // Estimate buffer size for efficiency

                // Efficiently construct the row string with pre-allocation
                rowStr += std::to_string(A.get_value(i, 0));
                for (size_t j = 1; j < cols; ++j) {
                    rowStr += " " + std::to_string(A.get_value(i, j));
                }
                rowStr += " " + std::to_string(b.get_value(i, 0)) + "\n";

                localRows.emplace_back(std::move(rowStr));
            }

            return localRows;
        }, startIdx, endIdx));
    }

    // Collect results and write to file synchronously
    for (auto& fut : futures) {
        std::vector<std::string> localRows = fut.get();
        for (const auto& rowStr : localRows) {
            fout << rowStr;
        }
    }

    fout.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Matrix A and vector b successfully written to " << filename << "\n";
    std::cout << "Writing took: " << std::chrono::duration<double>(end - start).count() << " seconds.\n";
}

/**
 * @brief Implementation of binary file writing with memory mapping
 * 
 * Uses memory mapping for high-performance binary file I/O. This method
 * is optimized for large matrices where traditional file I/O would be
 * a performance bottleneck.
 * 
 * Memory mapping benefits:
 * - Zero-copy operations for large files
 * - OS-level caching for optimal performance
 * - Reduced system call overhead
 * - Efficient for sequential access patterns
 * 
 * @tparam MatrixType Matrix storage type (dense or sparse)
 * @param A Coefficient matrix to write
 * @param b RHS vector to write
 * @param filename Output file path
 * @throws std::runtime_error on file access errors
 */
template <typename MatrixType>
void MatrixFileHandler::writeBinaryAB(const MatrixType& A, const MatrixType& b, const std::string& filename) {
    size_t rows = A.get_row();
    size_t cols = A.get_column();
    
    // Validate dimensions
    if (b.get_row() != rows || b.get_column() != 1) {
        std::cerr << "Dimension mismatch: Matrix A is " << rows << "x" << cols
                  << ", but vector b is " << b.get_row() << "x" << b.get_column() << "\n";
        throw std::invalid_argument("Dimension mismatch between A and b.");
    }

    // Calculate total file size
    size_t headerSize = sizeof(size_t) * 2;
    size_t matrixSize = sizeof(double) * rows * cols;
    size_t vectorSize = sizeof(double) * rows;
    size_t totalSize = headerSize + matrixSize + vectorSize;

    // Open the file
    int fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("Error opening file for writing");
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Resize the file to the required size
    if (ftruncate(fd, totalSize) == -1) {
        perror("Error resizing file");
        close(fd);
        throw std::runtime_error("Failed to resize file.");
    }

    // Memory map the file
    void* map = mmap(NULL, totalSize, PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        throw std::runtime_error("Failed to map file.");
    }

    // Write header
    size_t* ptr = static_cast<size_t*>(map);
    ptr[0] = rows;
    ptr[1] = cols;

    // Write matrix A
    double* matrixPtr = reinterpret_cast<double*>(ptr + 2); // Move past header
    for (size_t i = 0; i < rows * cols; ++i) {
        matrixPtr[i] = A.get_value(i / cols, i % cols);
    }

    // Write vector b
    double* vectorPtr = matrixPtr + (rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        vectorPtr[i] = b.get_value(i, 0);
    }

    // Sync to disk
    if (msync(map, totalSize, MS_SYNC) == -1) {
        perror("Could not sync the file to disk");
        // Proceeding even if msync fails, but you might want to handle it differently
    }

    // Unmap and close
    if (munmap(map, totalSize) == -1) {
        perror("Error unmapping the file");
        // Proceeding even if munmap fails
    }

    close(fd);

    std::cout << "Matrix A and vector b successfully written to " << filename << " using memory mapping.\n";
}

// Memory-mapped binary read function
template <typename MatrixType>
void MatrixFileHandler::loadBinaryAB(MatrixType& A, MatrixType& b, const std::string& filename) {
    // Open the file
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        throw std::runtime_error("Failed to open file for reading.");
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting the file size");
        close(fd);
        throw std::runtime_error("Failed to get file size.");
    }

    size_t totalSize = sb.st_size;
    if (totalSize < sizeof(size_t) * 2) {
        std::cerr << "File too small to contain header.\n";
        close(fd);
        throw std::runtime_error("Invalid file format.");
    }

    // Memory map the file
    void* map = mmap(NULL, totalSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        throw std::runtime_error("Failed to map file.");
    }

    // Read header
    size_t* ptr = static_cast<size_t*>(map);
    size_t rows = ptr[0];
    size_t cols = ptr[1];

    // Validate file size
    size_t expectedSize = sizeof(size_t) * 2 + sizeof(double) * rows * cols + sizeof(double) * rows;
    if (totalSize != expectedSize) {
        std::cerr << "File size does not match expected dimensions.\n";
        munmap(map, totalSize);
        close(fd);
        throw std::runtime_error("Invalid file size.");
    }

    // Initialize matrices
    A = MatrixType(rows, cols);
    b = MatrixType(rows, 1);

    // Read matrix A
    double* matrixPtr = reinterpret_cast<double*>(ptr + 2); // Move past header
    for (size_t i = 0; i < rows * cols; ++i) {
        A.set_value(i / cols, i % cols, matrixPtr[i]);
    }

    // Read vector b
    double* vectorPtr = matrixPtr + (rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        b.set_value(i, 0, vectorPtr[i]);
    }

    // Unmap and close
    if (munmap(map, totalSize) == -1) {
        perror("Error unmapping the file");
        // Proceeding even if munmap fails
    }

    close(fd);

    std::cout << "Matrix A and vector b successfully loaded from " << filename << " using memory mapping.\n";

    
}

} // namespace pnmatrix

#endif // MATRIX_FILE_HANDLER_H

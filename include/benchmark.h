#pragma once

#include "type.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <map>

namespace pnmatrix {
namespace benchmark {

/**
 * @brief High-resolution timer for performance measurement
 * 
 * Uses std::chrono::high_resolution_clock for precise timing.
 * Provides start/stop functionality and elapsed time calculation.
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;

public:
    Timer() : running_(false) {}

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }

    double elapsed_seconds() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time_);
        return duration.count() / 1e9;  // Convert nanoseconds to seconds
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

    void reset() {
        running_ = false;
    }
};

/**
 * @brief Statistics for benchmark results
 */
struct BenchmarkStats {
    double mean_time;
    double min_time;
    double max_time;
    double std_deviation;
    size_t iterations;
    std::vector<double> all_times;

    BenchmarkStats() : mean_time(0.0), min_time(0.0), max_time(0.0), std_deviation(0.0), iterations(0) {}
};

/**
 * @brief Benchmark result with metadata
 */
struct BenchmarkResult {
    std::string test_name;
    std::string matrix_type;
    size_type matrix_size;
    BenchmarkStats stats;
    std::string solver_name;

    BenchmarkResult(const std::string& name, const std::string& type, size_type size, const std::string& solver)
        : test_name(name), matrix_type(type), matrix_size(size), solver_name(solver) {}
};

/**
 * @brief Comprehensive benchmarking framework
 * 
 * Provides high-level interface for performance testing of matrix operations
 * and solvers with statistical analysis and result reporting.
 */
class BenchmarkSuite {
private:
    std::vector<BenchmarkResult> results_;
    std::string output_file_;

public:
    explicit BenchmarkSuite(const std::string& output_file = "benchmark_results.txt")
        : output_file_(output_file) {}

    /**
     * @brief Run a benchmark function multiple times and collect statistics
     * @tparam Func Function type to benchmark
     * @param test_name Name of the test
     * @param matrix_type Type of matrix (dense/sparse)
     * @param matrix_size Size of the matrix
     * @param solver_name Name of the solver being tested
     * @param func Function to benchmark
     * @param iterations Number of iterations to run
     * @return Benchmark statistics
     */
    template<typename Func>
    BenchmarkStats run_benchmark(const std::string& test_name,
                                 const std::string& matrix_type,
                                 size_type matrix_size,
                                 const std::string& solver_name,
                                 Func func,
                                 size_t iterations = 10) {
        std::vector<double> times;
        times.reserve(iterations);

        Timer timer;

        // Warm-up run (not counted in statistics)
        func();

        // Actual benchmark runs
        for (size_t i = 0; i < iterations; ++i) {
            timer.start();
            func();
            timer.stop();
            times.push_back(timer.elapsed_seconds());
            timer.reset();
        }

        // Calculate statistics
        BenchmarkStats stats = calculate_statistics(times);
        stats.iterations = iterations;
        stats.all_times = times;

        // Store result
        BenchmarkResult result(test_name, matrix_type, matrix_size, solver_name);
        result.stats = stats;
        results_.push_back(result);

        return stats;
    }

    /**
     * @brief Calculate statistical measures from timing data
     */
    BenchmarkStats calculate_statistics(const std::vector<double>& times) {
        BenchmarkStats stats;
        
        if (times.empty()) return stats;

        stats.all_times = times;
        stats.iterations = times.size();

        // Calculate mean
        stats.mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        // Calculate min and max
        stats.min_time = *std::min_element(times.begin(), times.end());
        stats.max_time = *std::max_element(times.begin(), times.end());

        // Calculate standard deviation
        double variance = 0.0;
        for (double time : times) {
            variance += (time - stats.mean_time) * (time - stats.mean_time);
        }
        variance /= times.size();
        stats.std_deviation = std::sqrt(variance);

        return stats;
    }

    /**
     * @brief Print benchmark results to console
     */
    void print_results() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARK RESULTS\n";
        std::cout << std::string(80, '=') << "\n\n";

        for (const auto& result : results_) {
            std::cout << "Test: " << result.test_name << "\n";
            std::cout << "Solver: " << result.solver_name << "\n";
            std::cout << "Matrix Type: " << result.matrix_type << "\n";
            std::cout << "Matrix Size: " << result.matrix_size << " x " << result.matrix_size << "\n";
            std::cout << "Iterations: " << result.stats.iterations << "\n";
            std::cout << "Mean Time: " << result.stats.mean_time << " seconds\n";
            std::cout << "Min Time: " << result.stats.min_time << " seconds\n";
            std::cout << "Max Time: " << result.stats.max_time << " seconds\n";
            std::cout << "Std Deviation: " << result.stats.std_deviation << " seconds\n";
            std::cout << std::string(40, '-') << "\n";
        }
    }

    /**
     * @brief Save benchmark results to file
     */
    void save_results() const {
        std::ofstream file(output_file_);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open output file " << output_file_ << "\n";
            return;
        }

        file << std::fixed << std::setprecision(6);
        file << "Test Name,Solver,Matrix Type,Matrix Size,Iterations,Mean Time (s),Min Time (s),Max Time (s),Std Deviation (s)\n";

        for (const auto& result : results_) {
            file << result.test_name << ","
                 << result.solver_name << ","
                 << result.matrix_type << ","
                 << result.matrix_size << ","
                 << result.stats.iterations << ","
                 << result.stats.mean_time << ","
                 << result.stats.min_time << ","
                 << result.stats.max_time << ","
                 << result.stats.std_deviation << "\n";
        }

        file.close();
        std::cout << "Results saved to " << output_file_ << "\n";
    }

    /**
     * @brief Clear all benchmark results
     */
    void clear_results() {
        results_.clear();
    }

    /**
     * @brief Get number of stored results
     */
    size_t result_count() const {
        return results_.size();
    }

    /**
     * @brief Compare performance between two solvers
     */
    void compare_solvers(const std::string& solver1, const std::string& solver2) const {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "SOLVER COMPARISON: " << solver1 << " vs " << solver2 << "\n";
        std::cout << std::string(60, '=') << "\n";

        for (const auto& result : results_) {
            if (result.solver_name == solver1 || result.solver_name == solver2) {
                std::cout << result.test_name << " (" << result.matrix_size << "x" << result.matrix_size << "): ";
                std::cout << result.solver_name << " = " << result.stats.mean_time << "s\n";
            }
        }
    }

    /**
     * @brief Generate performance summary report
     */
    void generate_summary() const {
        if (results_.empty()) {
            std::cout << "No benchmark results available.\n";
            return;
        }

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "PERFORMANCE SUMMARY\n";
        std::cout << std::string(60, '=') << "\n";

        // Group by solver
        std::map<std::string, std::vector<const BenchmarkResult*>> solver_results_map;
        for (const auto& result : results_) {
            solver_results_map[result.solver_name].push_back(&result);
        }

        // Report per solver
        for (const auto& solver_pair : solver_results_map) {
            const std::string& solver = solver_pair.first;
            const std::vector<const BenchmarkResult*>& results = solver_pair.second;
            
            double total_time = 0.0;
            size_t total_operations = 0;

            for (const auto* result : results) {
                total_time += result->stats.mean_time;
                total_operations += result->matrix_size * result->matrix_size;
            }

            std::cout << solver << ":\n";
            std::cout << "  Tests run: " << results.size() << "\n";
            std::cout << "  Total time: " << total_time << " seconds\n";
            std::cout << "  Avg time per test: " << total_time / results.size() << " seconds\n";
            std::cout << "\n";
        }
    }
};

/**
 * @brief Convenience functions for common benchmark scenarios
 */
namespace utils {

/**
 * @brief Benchmark matrix multiplication
 */
template<typename MatrixType>
void benchmark_matrix_multiplication(BenchmarkSuite& suite, 
                                     const std::vector<size_type>& sizes,
                                     size_t iterations = 5) {
    for (auto size : sizes) {
        suite.run_benchmark(
            "Matrix Multiplication",
            "Dense",
            size,
            "Native",
            [size]() {
                MatrixType A(size, size);
                MatrixType B(size, size);
                
                // Initialize with random values
                for (size_type i = 0; i < size; ++i) {
                    for (size_type j = 0; j < size; ++j) {
                        A.set_value(i, j, (i + j) % 100 / 100.0);
                        B.set_value(i, j, (i * j) % 100 / 100.0);
                    }
                }
                
                MatrixType C = A * B;  // Matrix multiplication
                volatile auto result = C.get_value(0, 0);  // Prevent optimization
            },
            iterations
        );
    }
}

/**
 * @brief Benchmark solver performance
 */
template<typename MatrixType, typename SolverType>
void benchmark_solver(BenchmarkSuite& suite,
                     const std::string& solver_name,
                     const std::vector<size_type>& sizes,
                     size_t iterations = 3) {
    for (auto size : sizes) {
        suite.run_benchmark(
            "Linear System Solve",
            "Dense",
            size,
            solver_name,
            [size, solver_name]() {
                MatrixType A(size, size);
                MatrixType b(size, 1);
                
                // Create diagonally dominant matrix (guaranteed convergence)
                for (size_type i = 0; i < size; ++i) {
                    for (size_type j = 0; j < size; ++j) {
                        if (i == j) {
                            A.set_value(i, j, size + 1.0);  // Diagonal dominance
                        } else {
                            A.set_value(i, j, 1.0);
                        }
                    }
                    b.set_value(i, 0, i + 1.0);
                }
                
                typename SolverType::option opts;
                SolverType solver(opts);
                MatrixType x = solver.solve(A, b);
                volatile auto result = x.get_value(0, 0);  // Prevent optimization
            },
            iterations
        );
    }
}

} // namespace utils
} // namespace benchmark
} // namespace pnmatrix

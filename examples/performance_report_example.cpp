/**
 * @file performance_report_example.cpp
 * @brief Example demonstrating comprehensive performance reporting
 * 
 * This example shows how to use the performance reporting system to:
 * - Run benchmarks for different solvers and matrix sizes
 * - Generate comprehensive HTML and text reports
 * - Export data for external analysis
 * - Perform statistical analysis and comparisons
 */

#include "../include/performance_reporter.h"
#include "../include/matrix.h"
#include "../include/dense_matrix_storage.h"
#include "../include/sparse_matrix_storage.h"
#include "../include/gauss_seidel_solver.h"
#include "../include/gaussian_elimination.h"
#include "../include/jacobian_solver.h"
#include "../include/gmres_solver.h"
#include "matrix_file_handler.h"
#include "matrix_solver.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace pnmatrix;
using namespace pnmatrix::benchmark;
using DenseMatrix = matrix<dense_matrix_storage<double>>;
using SparseMatrix = matrix<sparse_matrix_storage<double>>;

/**
 * @brief Run comprehensive benchmark suite with multiple solvers
 * 
 * @param reporter Performance reporter to collect results
 * @param sizes Vector of matrix sizes to test
 * @param iterations Number of iterations per test
 */
void run_comprehensive_benchmark(PerformanceReporter& reporter, 
                                 const std::vector<size_t>& sizes, 
                                 int iterations) {
    std::cout << "Running comprehensive benchmark suite...\n";
    std::cout << "Matrix sizes: ";
    for (size_t size : sizes) std::cout << size << " ";
    std::cout << "\nIterations per test: " << iterations << "\n\n";
    
    for (size_t size : sizes) {
        std::cout << "Testing matrix size: " << size << "x" << size << "\n";
        
        for (int iter = 0; iter < iterations; ++iter) {
            std::string filename = "benchmark_" + std::to_string(size) + "_" + std::to_string(iter) + ".dat";
            
            // Test Gauss-Seidel
            {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Generate matrix
                DenseMatrix A(size, size);
                DenseMatrix x(size, 1);
                DenseMatrix b(size, 1);
                std::random_device rd;
                generateRandomMatrix(A, x, b, size, rd);
                
                auto gen_end = std::chrono::high_resolution_clock::now();
                double generation_time = std::chrono::duration<double>(gen_end - start).count();
                
                // Solve with Gauss-Seidel
                gauss_seidel::option op;
                op.rm = 1e-6;
                gauss_seidel solver(op);
                
                auto solve_start = std::chrono::high_resolution_clock::now();
                auto result = solver.solve(A, b);
                auto solve_end = std::chrono::high_resolution_clock::now();
                double solve_time = std::chrono::duration<double>(solve_end - solve_start).count();
                
                // Calculate final error
                double final_error = 0.0;
                for (size_t i = 0; i < size; ++i) {
                    double ax_i = 0.0;
                    for (size_t j = 0; j < size; ++j) {
                        ax_i += A.get_value(i, j) * result.get_value(j, 0);
                    }
                    double diff = ax_i - b.get_value(i, 0);
                    final_error += diff * diff;
                }
                final_error = std::sqrt(final_error);
                
                BenchmarkResult result_data;
                result_data.solver_name = "Gauss-Seidel";
                result_data.matrix_type = "Dense";
                result_data.matrix_size = size;
                result_data.generation_time = generation_time;
                result_data.load_time = 0.0; // Not loading from file in this example
                result_data.solve_time = solve_time;
                result_data.total_time = generation_time + solve_time;
                result_data.iterations = solver.get_iteration_count();
                result_data.final_error = final_error;
                result_data.converged = true;
                result_data.timestamp = std::chrono::system_clock::now();
                
                reporter.add_result(result_data);
            }
            
            // Test Gaussian Elimination
            {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Generate matrix
                DenseMatrix A(size, size);
                DenseMatrix x(size, 1);
                DenseMatrix b(size, 1);
                std::random_device rd;
                generateRandomMatrix(A, x, b, size, rd);
                
                auto gen_end = std::chrono::high_resolution_clock::now();
                double generation_time = std::chrono::duration<double>(gen_end - start).count();
                
                // Solve with Gaussian Elimination
                gaussian_elimination::option op;
                op.rm = 1e-6;
                gaussian_elimination solver(op);
                
                auto solve_start = std::chrono::high_resolution_clock::now();
                auto result = solver.solve(A, b);
                auto solve_end = std::chrono::high_resolution_clock::now();
                double solve_time = std::chrono::duration<double>(solve_end - solve_start).count();
                
                // Calculate final error
                double final_error = 0.0;
                for (size_t i = 0; i < size; ++i) {
                    double ax_i = 0.0;
                    for (size_t j = 0; j < size; ++j) {
                        ax_i += A.get_value(i, j) * result.get_value(j, 0);
                    }
                    double diff = ax_i - b.get_value(i, 0);
                    final_error += diff * diff;
                }
                final_error = std::sqrt(final_error);
                
                BenchmarkResult result_data;
                result_data.solver_name = "Gaussian Elimination";
                result_data.matrix_type = "Dense";
                result_data.matrix_size = size;
                result_data.generation_time = generation_time;
                result_data.load_time = 0.0;
                result_data.solve_time = solve_time;
                result_data.total_time = generation_time + solve_time;
                result_data.iterations = 0; // Direct method
                result_data.final_error = final_error;
                result_data.converged = true;
                result_data.timestamp = std::chrono::system_clock::now();
                
                reporter.add_result(result_data);
            }
            
            // Test Jacobi (if size is reasonable)
            if (size <= 500) {
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    // Generate matrix
                    DenseMatrix A(size, size);
                    DenseMatrix x(size, 1);
                    DenseMatrix b(size, 1);
                    std::random_device rd;
                    generateRandomMatrix(A, x, b, size, rd);
                    
                    auto gen_end = std::chrono::high_resolution_clock::now();
                    double generation_time = std::chrono::duration<double>(gen_end - start).count();
                    
                    // Solve with Jacobi
                    jacobian::option op;
                    op.rm = 1e-6;
                    jacobian solver(op);
                    
                    auto solve_start = std::chrono::high_resolution_clock::now();
                    auto result = solver.solve(A, b);
                    auto solve_end = std::chrono::high_resolution_clock::now();
                    double solve_time = std::chrono::duration<double>(solve_end - solve_start).count();
                    
                    // Calculate final error
                    double final_error = 0.0;
                    for (size_t i = 0; i < size; ++i) {
                        double ax_i = 0.0;
                        for (size_t j = 0; j < size; ++j) {
                            ax_i += A.get_value(i, j) * result.get_value(j, 0);
                        }
                        double diff = ax_i - b.get_value(i, 0);
                        final_error += diff * diff;
                    }
                    final_error = std::sqrt(final_error);
                    
                    BenchmarkResult result_data;
                    result_data.solver_name = "Jacobi";
                    result_data.matrix_type = "Dense";
                    result_data.matrix_size = size;
                    result_data.generation_time = generation_time;
                    result_data.load_time = 0.0;
                    result_data.solve_time = solve_time;
                    result_data.total_time = generation_time + solve_time;
                    result_data.iterations = solver.get_iteration_count();
                    result_data.final_error = final_error;
                    result_data.converged = true;
                    result_data.timestamp = std::chrono::system_clock::now();
                    
                    reporter.add_result(result_data);
                }
            }
            
            if (iter == 0) {
                std::cout << "  Completed iteration " << (iter + 1) << "/" << iterations << "\n";
            }
        }
    }
}

/**
 * @brief Generate all report formats
 * 
 * @param reporter Performance reporter with collected data
 * @param base_filename Base filename for reports
 */
void generate_all_reports(PerformanceReporter& reporter, const std::string& base_filename) {
    std::cout << "\nGenerating performance reports...\n";
    
    try {
        // Generate HTML report
        reporter.generate_html_report(base_filename + ".html", "Matrix Solver Performance Report");
        std::cout << "✓ HTML report generated: " << base_filename << ".html\n";
        
        // Generate text report
        reporter.generate_text_report(base_filename + ".txt", "Matrix Solver Performance Report");
        std::cout << "✓ Text report generated: " << base_filename << ".txt\n";
        
        // Generate CSV export
        reporter.generate_csv_export(base_filename + ".csv");
        std::cout << "✓ CSV export generated: " << base_filename << ".csv\n";
        
        // Print summary to console
        auto comparison = reporter.get_solver_comparison();
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "PERFORMANCE SUMMARY\n";
        std::cout << std::string(50, '=') << "\n";
        std::cout << "Total benchmarks: " << reporter.get_result_count() << "\n";
        std::cout << "Fastest solver: " << comparison.fastest_solver << "\n";
        std::cout << "Most accurate: " << comparison.most_accurate_solver << "\n";
        std::cout << "Most consistent: " << comparison.most_consistent_solver << "\n";
        
        std::cout << "\nPerformance Ranking:\n";
        auto ranking = comparison.get_ranking_by_time();
        for (size_t i = 0; i < ranking.size(); ++i) {
            const auto& stats = comparison.solver_stats.at(ranking[i]);
            std::cout << (i + 1) << ". " << ranking[i] 
                      << " - " << std::fixed << std::setprecision(3) << stats.mean << "s"
                      << " (±" << std::setprecision(2) << stats.std_dev << "s)\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating reports: " << e.what() << "\n";
    }
}

/**
 * @brief Demonstrate scalability analysis
 * 
 * @param reporter Performance reporter with collected data
 */
void demonstrate_scalability_analysis(PerformanceReporter& reporter) {
    std::cout << "\n" << std::string(40, '=') << "\n";
    std::cout << "SCALABILITY ANALYSIS\n";
    std::cout << std::string(40, '=') << "\n";
    
    auto scalability = reporter.get_scalability_analysis();
    
    for (const auto& [solver, data] : scalability) {
        std::cout << solver << ":\n";
        std::cout << "  Size -> Time (s)\n";
        for (const auto& [size, time] : data) {
            std::cout << "  " << std::setw(4) << size << " -> " 
                      << std::fixed << std::setprecision(3) << time << "\n";
        }
        std::cout << "\n";
        
        // Calculate growth rate
        if (data.size() >= 2) {
            double first_time = data[0].second;
            double last_time = data.back().second;
            size_t first_size = data[0].first;
            size_t last_size = data.back().first;
            
            if (first_time > 0 && last_size > first_size) {
                double size_growth = static_cast<double>(last_size) / first_size;
                double time_growth = last_time / first_time;
                double complexity = std::log(time_growth) / std::log(size_growth);
                
                std::cout << "  Estimated complexity: O(n^" << std::setprecision(2) << complexity << ")\n";
            }
        }
        std::cout << "\n";
    }
}

/**
 * @brief Main function demonstrating performance reporting
 */
int main() {
    std::cout << "=== Matrix Solver Performance Reporting Example ===\n\n";
    
    // Create performance reporter
    PerformanceReporter reporter;
    
    // Define test matrix sizes
    std::vector<size_t> sizes = {100, 200, 500, 1000};
    int iterations = 3;
    
    // Run comprehensive benchmark
    run_comprehensive_benchmark(reporter, sizes, iterations);
    
    // Generate all reports
    generate_all_reports(reporter, "matrix_performance_report");
    
    // Demonstrate scalability analysis
    demonstrate_scalability_analysis(reporter);
    
    std::cout << "\n=== Performance Reporting Complete ===\n";
    std::cout << "Open the HTML report in your browser to view interactive charts and detailed analysis.\n";
    
    return 0;
}

/**
 * @file performance_reporter.h
 * @brief Comprehensive performance reporting system for matrix solver benchmarks
 * 
 * This header provides a complete performance analysis and reporting framework
 * for the matrix solver benchmarking system. It includes:
 * 
 * - Statistical analysis of solver performance
 * - HTML and text report generation
 * - Performance comparison and ranking
 * - Visualization data export
 * - Convergence analysis for iterative solvers
 * - Memory usage tracking
 * - Scalability analysis
 */

#pragma once

#include <set>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <memory>

namespace pnmatrix {
namespace benchmark {

/**
 * @brief Individual benchmark result data structure
 * 
 * Stores all relevant performance metrics for a single benchmark run,
 * including timing data, solver information, and convergence metrics.
 */
struct BenchmarkResult {
    std::string solver_name;           ///< Name of the solver (e.g., "Gauss-Seidel", "Gaussian Elimination")
    std::string matrix_type;           ///< Matrix storage type (e.g., "Dense", "Sparse")
    size_t matrix_size;                 ///< Matrix dimension (n x n)
    double generation_time;            ///< Time to generate matrix (seconds)
    double load_time;                   ///< Time to load matrix from file (seconds)
    double solve_time;                  ///< Time to solve system (seconds)
    double total_time;                  ///< Total execution time (seconds)
    size_t iterations;                  ///< Number of iterations (for iterative solvers)
    double final_error;                 ///< Final relative error
    bool converged;                    ///< Whether solver converged (iterative solvers)
    std::chrono::system_clock::time_point timestamp; ///< When benchmark was run
    
    /**
     * @brief Calculate performance metrics (operations per second)
     * 
     * @return Operations per second for this benchmark
     */
    double get_ops_per_second() const {
        // Approximate operation count: n^3 for direct methods, n^2*iterations for iterative
        if (solver_name.find("Gauss-Seidel") != std::string::npos || 
            solver_name.find("Jacobi") != std::string::npos) {
            return (matrix_size * matrix_size * iterations) / solve_time;
        } else {
            return (matrix_size * matrix_size * matrix_size) / solve_time; // n^3 for direct methods
        }
    }
};

/**
 * @brief Statistical summary for multiple benchmark runs
 * 
 * Provides statistical analysis including mean, median, standard deviation,
 * and confidence intervals for performance metrics.
 */
struct StatisticalSummary {
    double mean;                        ///< Mean value
    double median;                      ///< Median value
    double std_dev;                      ///< Standard deviation
    double min_val;                      ///< Minimum value
    double max_val;                      ///< Maximum value
    double confidence_95_low;            ///< 95% confidence interval lower bound
    double confidence_95_high;           ///< 95% confidence interval upper bound
    size_t sample_count;                 ///< Number of samples
    
    /**
     * @brief Calculate coefficient of variation
     * 
     * @return Coefficient of variation (std_dev / mean)
     */
    double get_coefficient_of_variation() const {
        return mean > 0 ? std_dev / mean : 0.0;
    }
};

/**
 * @brief Performance comparison between different solvers
 * 
 * Ranks solvers by various performance metrics and provides
 * comparative analysis.
 */
struct SolverComparison {
    std::map<std::string, StatisticalSummary> solver_stats; ///< Solver name -> statistics
    std::string fastest_solver;                                   ///< Fastest solver by total time
    std::string most_accurate_solver;                             ///< Most accurate solver
    std::string most_consistent_solver;                           ///< Most consistent (lowest CV)
    
    /**
     * @brief Get performance ranking by total time
     * 
     * @return Vector of solver names ranked by performance (fastest first)
     */
    std::vector<std::string> get_ranking_by_time() const {
        std::vector<std::pair<std::string, double>> rankings;
        for (const auto& [solver, stats] : solver_stats) {
            rankings.emplace_back(solver, stats.mean);
        }
        std::sort(rankings.begin(), rankings.end(), 
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        std::vector<std::string> result;
        for (const auto& [solver, _] : rankings) {
            result.push_back(solver);
        }
        return result;
    }
};

/**
 * @brief Comprehensive performance reporter
 * 
 * Main class for generating performance reports from benchmark data.
 * Supports multiple output formats and provides detailed analysis.
 */
class PerformanceReporter {
public:
    /**
     * @brief Add a benchmark result to the reporter
     * 
     * @param result Benchmark result to add
     */
    void add_result(const BenchmarkResult& result);
    
    /**
     * @brief Generate HTML performance report
     * 
     * Creates a comprehensive HTML report with charts, tables, and analysis.
     * 
     * @param filename Output file path
     * @param title Report title
     */
    void generate_html_report(const std::string& filename, 
                               const std::string& title = "Matrix Solver Performance Report");
    
    /**
     * @brief Generate text performance report
     * 
     * Creates a plain text report with statistical analysis and comparisons.
     * 
     * @param filename Output file path
     * @param title Report title
     */
    void generate_text_report(const std::string& filename, 
                               const std::string& title = "Matrix Solver Performance Report");
    
    /**
     * @brief Generate CSV data for external analysis
     * 
     * Exports benchmark data in CSV format for use with spreadsheet
     * applications or data analysis tools.
     * 
     * @param filename Output file path
     */
    void generate_csv_export(const std::string& filename);
    
    /**
     * @brief Get solver comparison analysis
     * 
     * @return SolverComparison object with detailed analysis
     */
    SolverComparison get_solver_comparison();
    
    /**
     * @brief Get scalability analysis
     * 
     * Analyzes how solver performance scales with matrix size.
     * 
     * @return Map of solver names to size vs time data
     */
    std::map<std::string, std::vector<std::pair<size_t, double>>> get_scalability_analysis();
    
    /**
     * @brief Clear all benchmark results
     */
    void clear_results();
    
    /**
     * @brief Get number of benchmark results
     * 
     * @return Number of stored results
     */
    size_t get_result_count() const { return results_.size(); }

private:
    std::vector<BenchmarkResult> results_;
    
    // Helper methods
    StatisticalSummary calculate_statistics(const std::vector<double>& values);
    std::string generate_html_header(const std::string& title);
    std::string generate_html_footer();
    std::string generate_performance_table();
    std::string generate_charts_section();
    std::string generate_analysis_section();
    std::string generate_scalability_chart_data();
    std::string format_time(double seconds);
    std::string format_number(double value, int precision = 2);
};

/**
 * @brief Add benchmark result to the reporter
 * 
 * @param result Benchmark result to add
 */
void PerformanceReporter::add_result(const BenchmarkResult& result) {
    results_.push_back(result);
}

/**
 * @brief Calculate statistical summary for a set of values
 * 
 * @param values Vector of numeric values
 * @return StatisticalSummary with calculated statistics
 */
StatisticalSummary PerformanceReporter::calculate_statistics(const std::vector<double>& values) {
    StatisticalSummary stats;
    
    if (values.empty()) return stats;
    
    stats.sample_count = values.size();
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    // Calculate median
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    if (sorted_values.size() % 2 == 0) {
        stats.median = (sorted_values[sorted_values.size()/2 - 1] + sorted_values[sorted_values.size()/2]) / 2.0;
    } else {
        stats.median = sorted_values[sorted_values.size()/2];
    }
    
    stats.min_val = sorted_values.front();
    stats.max_val = sorted_values.back();
    
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (double val : values) {
        sum_sq_diff += (val - stats.mean) * (val - stats.mean);
    }
    stats.std_dev = std::sqrt(sum_sq_diff / values.size());
    
    // Calculate 95% confidence interval
    double margin = 1.96 * stats.std_dev / std::sqrt(values.size());
    stats.confidence_95_low = stats.mean - margin;
    stats.confidence_95_high = stats.mean + margin;
    
    return stats;
}

/**
 * @brief Generate HTML performance report
 * 
 * @param filename Output file path
 * @param title Report title
 */
void PerformanceReporter::generate_html_report(const std::string& filename, const std::string& title) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << generate_html_header(title);
    file << generate_performance_table();
    file << generate_charts_section();
    file << generate_analysis_section();
    file << generate_html_footer();
    
    file.close();
}

/**
 * @brief Generate text performance report
 * 
 * @param filename Output file path
 * @param title Report title
 */
void PerformanceReporter::generate_text_report(const std::string& filename, const std::string& title) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << title << "\n";
    file << std::string(title.length(), '=') << "\n\n";
    
    // Summary statistics
    auto comparison = get_solver_comparison();
    file << "SOLVER COMPARISON SUMMARY\n";
    file << std::string(25, '-') << "\n";
    
    for (const auto& [solver, stats] : comparison.solver_stats) {
        file << solver << ":\n";
        file << "  Mean Time: " << format_time(stats.mean) << "\n";
        file << "  Std Dev: " << format_time(stats.std_dev) << "\n";
        file << "  Min Time: " << format_time(stats.min_val) << "\n";
        file << "  Max Time: " << format_time(stats.max_val) << "\n";
        file << "  95% CI: [" << format_time(stats.confidence_95_low) << ", " 
             << format_time(stats.confidence_95_high) << "]\n";
        file << "  Coeff. of Variation: " << format_number(stats.get_coefficient_of_variation() * 100, 2) << "%\n\n";
    }
    
    // Performance ranking
    auto ranking = comparison.get_ranking_by_time();
    file << "PERFORMANCE RANKING (Fastest to Slowest)\n";
    file << std::string(35, '-') << "\n";
    for (size_t i = 0; i < ranking.size(); ++i) {
        const auto& stats = comparison.solver_stats.at(ranking[i]);
        file << (i + 1) << ". " << ranking[i] << " - " << format_time(stats.mean) << "\n";
    }
    
    file.close();
}

/**
 * @brief Generate CSV export
 * 
 * @param filename Output file path
 */
void PerformanceReporter::generate_csv_export(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // CSV header
    file << "Solver,Matrix Type,Matrix Size,Generation Time,Load Time,Solve Time,Total Time,Iterations,Final Error,Converged,Timestamp\n";
    
    // Data rows
    for (const auto& result : results_) {
        file << result.solver_name << ","
             << result.matrix_type << ","
             << result.matrix_size << ","
             << result.generation_time << ","
             << result.load_time << ","
             << result.solve_time << ","
             << result.total_time << ","
             << result.iterations << ","
             << result.final_error << ","
             << (result.converged ? "true" : "false") << ","
             << std::chrono::duration_cast<std::chrono::seconds>(result.timestamp.time_since_epoch()).count() << "\n";
    }
    
    file.close();
}

/**
 * @brief Get solver comparison analysis
 * 
 * @return SolverComparison object with detailed analysis
 */
SolverComparison PerformanceReporter::get_solver_comparison() {
    SolverComparison comparison;
    
    // Group results by solver
    std::map<std::string, std::vector<double>> total_times;
    std::map<std::string, std::vector<double>> solve_times;
    std::map<std::string, std::vector<double>> errors;
    
    for (const auto& result : results_) {
        total_times[result.solver_name].push_back(result.total_time);
        solve_times[result.solver_name].push_back(result.solve_time);
        errors[result.solver_name].push_back(result.final_error);
    }
    
    // Calculate statistics for each solver
    for (const auto& [solver, times] : total_times) {
        comparison.solver_stats[solver] = calculate_statistics(times);
    }
    
    // Find fastest solver
    if (!comparison.solver_stats.empty()) {
        comparison.fastest_solver = std::min_element(
            comparison.solver_stats.begin(), comparison.solver_stats.end(),
            [](const auto& a, const auto& b) { return a.second.mean < b.second.mean; }
        )->first;
    }
    
    // Find most accurate solver (lowest error)
    std::map<std::string, double> avg_errors;
    for (const auto& [solver, error_vals] : errors) {
        if (!error_vals.empty()) {
            avg_errors[solver] = std::accumulate(error_vals.begin(), error_vals.end(), 0.0) / error_vals.size();
        }
    }
    if (!avg_errors.empty()) {
        comparison.most_accurate_solver = std::min_element(
            avg_errors.begin(), avg_errors.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        )->first;
    }
    
    // Find most consistent solver (lowest coefficient of variation)
    if (!comparison.solver_stats.empty()) {
        comparison.most_consistent_solver = std::min_element(
            comparison.solver_stats.begin(), comparison.solver_stats.end(),
            [](const auto& a, const auto& b) { 
                return a.second.get_coefficient_of_variation() < b.second.get_coefficient_of_variation(); 
            }
        )->first;
    }
    
    return comparison;
}

/**
 * @brief Get scalability analysis
 * 
 * @return Map of solver names to size vs time data
 */
std::map<std::string, std::vector<std::pair<size_t, double>>> PerformanceReporter::get_scalability_analysis() {
    std::map<std::string, std::vector<std::pair<size_t, double>>> scalability;
    
    // Group results by solver and matrix size
    std::map<std::string, std::map<size_t, std::vector<double>>> solver_size_times;
    
    for (const auto& result : results_) {
        solver_size_times[result.solver_name][result.matrix_size].push_back(result.total_time);
    }
    
    // Calculate average time for each size
    for (const auto& [solver, size_map] : solver_size_times) {
        for (const auto& [size, times] : size_map) {
            double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            scalability[solver].emplace_back(size, avg_time);
        }
    }
    
    // Sort by matrix size for each solver
    for (auto& [solver, data] : scalability) {
        std::sort(data.begin(), data.end(), 
                 [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    
    return scalability;
}

/**
 * @brief Clear all benchmark results
 */
void PerformanceReporter::clear_results() {
    results_.clear();
}

/**
 * @brief Generate HTML header
 * 
 * @param title Report title
 * @return HTML header string
 */
std::string PerformanceReporter::generate_html_header(const std::string& title) {
    std::stringstream html;
    html << R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>)" << title << R"(</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f8f9fa; font-weight: bold; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .metric { text-align: right; font-family: 'Courier New', monospace; }
        .fastest { background-color: #d4edda !important; }
        .slowest { background-color: #f8d7da !important; }
        .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 4px; }
        .summary-box { background: #e9ecef; padding: 15px; border-radius: 4px; margin: 10px 0; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>)" << title << R"(</h1>
)";
    return html.str();
}

/**
 * @brief Generate HTML footer
 * 
 * @return HTML footer string
 */
std::string PerformanceReporter::generate_html_footer() {
    std::stringstream html;
    html << R"(
    </div>
</body>
</html>)";
    return html.str();
}

/**
 * @brief Generate performance table
 * 
 * @return HTML table string
 */
std::string PerformanceReporter::generate_performance_table() {
    std::stringstream html;
    
    auto comparison = get_solver_comparison();
    
    html << "<h2>Performance Summary</h2>\n";
    html << "<table>\n";
    html << "<tr><th>Solver</th><th>Avg Total Time</th><th>Std Dev</th><th>Min Time</th><th>Max Time</th><th>95% CI</th><th>CV</th></tr>\n";
    
    auto ranking = comparison.get_ranking_by_time();
    for (const auto& solver : ranking) {
        const auto& stats = comparison.solver_stats.at(solver);
        std::string row_class = "";
        if (solver == ranking.front()) row_class = " class=\"fastest\"";
        if (solver == ranking.back()) row_class = " class=\"slowest\"";
        
        html << "<tr" << row_class << ">\n";
        html << "<td>" << solver << "</td>\n";
        html << "<td class=\"metric\">" << format_time(stats.mean) << "</td>\n";
        html << "<td class=\"metric\">" << format_time(stats.std_dev) << "</td>\n";
        html << "<td class=\"metric\">" << format_time(stats.min_val) << "</td>\n";
        html << "<td class=\"metric\">" << format_time(stats.max_val) << "</td>\n";
        html << "<td class=\"metric\">[" << format_time(stats.confidence_95_low) 
             << ", " << format_time(stats.confidence_95_high) << "]</td>\n";
        html << "<td class=\"metric\">" << format_number(stats.get_coefficient_of_variation() * 100, 2) << "%</td>\n";
        html << "</tr>\n";
    }
    
    html << "</table>\n";
    
    return html.str();
}

/**
 * @brief Generate charts section
 * 
 * @return HTML charts section string
 */
std::string PerformanceReporter::generate_charts_section() {
    std::stringstream html;
    
    html << "<h2>Performance Charts</h2>\n";
    
    // Performance comparison chart
    html << "<div class=\"chart-container\">\n";
    html << "<h3>Solver Performance Comparison</h3>\n";
    html << "<canvas id=\"performanceChart\" width=\"400\" height=\"200\"></canvas>\n";
    html << "</div>\n";
    
    // Scalability chart
    html << "<div class=\"chart-container\">\n";
    html << "<h3>Scalability Analysis</h3>\n";
    html << "<canvas id=\"scalabilityChart\" width=\"400\" height=\"200\"></canvas>\n";
    html << "</div>\n";
    
    // Chart JavaScript
    html << "<script>\n";
    
    // Performance comparison chart
    auto comparison = get_solver_comparison();
    html << "const performanceCtx = document.getElementById('performanceChart').getContext('2d');\n";
    html << "const performanceChart = new Chart(performanceCtx, {\n";
    html << "    type: 'bar',\n";
    html << "    data: {\n";
    html << "        labels: [";
    bool first = true;
    for (const auto& solver : comparison.get_ranking_by_time()) {
        if (!first) html << ", ";
        html << "'" << solver << "'";
        first = false;
    }
    html << "],\n";
    html << "        datasets: [{\n";
    html << "            label: 'Average Total Time (s)',\n";
    html << "            data: [";
    first = true;
    for (const auto& solver : comparison.get_ranking_by_time()) {
        if (!first) html << ", ";
        html << comparison.solver_stats.at(solver).mean;
        first = false;
    }
    html << "],\n";
    html << "            backgroundColor: 'rgba(54, 162, 235, 0.8)',\n";
    html << "            borderColor: 'rgba(54, 162, 235, 1)',\n";
    html << "            borderWidth: 1\n";
    html << "        }]\n";
    html << "    },\n";
    html << "    options: {\n";
    html << "        responsive: true,\n";
    html << "        scales: {\n";
    html << "            y: {\n";
    html << "                beginAtZero: true,\n";
    html << "                title: {\n";
    html << "                    display: true,\n";
    html << "                    text: 'Time (seconds)'\n";
    html << "                }\n";
    html << "            }\n";
    html << "        }\n";
    html << "    }\n";
    html << "});\n";
    
    // Scalability chart
    auto scalability = get_scalability_analysis();
    html << "const scalabilityCtx = document.getElementById('scalabilityChart').getContext('2d');\n";
    html << "const scalabilityChart = new Chart(scalabilityCtx, {\n";
    html << "    type: 'line',\n";
    html << "    data: {\n";
    html << "        labels: [";
    std::set<size_t> all_sizes;
    for (const auto& [solver, data] : scalability) {
        for (const auto& [size, _] : data) {
            all_sizes.insert(size);
        }
    }
    first = true;
    for (size_t size : all_sizes) {
        if (!first) html << ", ";
        html << size;
        first = false;
    }
    html << "],\n";
    html << "        datasets: [";
    first = true;
    for (const auto& [solver, data] : scalability) {
        if (!first) html << ", ";
        html << "{\n";
        html << "            label: '" << solver << "',\n";
        html << "            data: [";
        bool first_data = true;
        for (size_t size : all_sizes) {
            auto it = std::find_if(data.begin(), data.end(), 
                                  [size](const auto& pair) { return pair.first == size; });
            if (!first_data) html << ", ";
            if (it != data.end()) {
                html << it->second;
            } else {
                html << "null";
            }
            first_data = false;
        }
        html << "],\n";
        html << "            borderColor: 'rgba(" << (std::hash<std::string>{}(solver) % 255) << ", 99, 132, 1)',\n";
        html << "            backgroundColor: 'rgba(" << (std::hash<std::string>{}(solver) % 255) << ", 99, 132, 0.2)',\n";
        html << "            tension: 0.1\n";
        html << "        }";
        first = false;
    }
    html << "]\n";
    html << "    },\n";
    html << "    options: {\n";
    html << "        responsive: true,\n";
    html << "        scales: {\n";
    html << "            y: {\n";
    html << "                beginAtZero: true,\n";
    html << "                title: {\n";
    html << "                    display: true,\n";
    html << "                    text: 'Time (seconds)'\n";
    html << "                }\n";
    html << "            },\n";
    html << "            x: {\n";
    html << "                title: {\n";
    html << "                    display: true,\n";
    html << "                    text: 'Matrix Size'\n";
    html << "                }\n";
    html << "            }\n";
    html << "        }\n";
    html << "    }\n";
    html << "});\n";
    
    html << "</script>\n";
    
    return html.str();
}

/**
 * @brief Generate analysis section
 * 
 * @return HTML analysis section string
 */
std::string PerformanceReporter::generate_analysis_section() {
    std::stringstream html;
    
    auto comparison = get_solver_comparison();
    
    html << "<h2>Performance Analysis</h2>\n";
    
    // Summary boxes
    html << "<div class=\"summary-box\">\n";
    html << "<h3>Key Findings</h3>\n";
    html << "<p><strong>Fastest Solver:</strong> " << comparison.fastest_solver << "</p>\n";
    html << "<p><strong>Most Accurate:</strong> " << comparison.most_accurate_solver << "</p>\n";
    html << "<p><strong>Most Consistent:</strong> " << comparison.most_consistent_solver << "</p>\n";
    html << "</div>\n";
    
    // Detailed analysis
    html << "<div class=\"summary-box\">\n";
    html << "<h3>Performance Characteristics</h3>\n";
    
    for (const auto& [solver, stats] : comparison.solver_stats) {
        html << "<h4>" << solver << "</h4>\n";
        html << "<ul>\n";
        html << "<li>Average execution time: " << format_time(stats.mean) << "</li>\n";
        html << "<li>Standard deviation: " << format_time(stats.std_dev) << "</li>\n";
        html << "<li>Performance range: " << format_time(stats.min_val) << " to " << format_time(stats.max_val) << "</li>\n";
        html << "<li>Consistency (CV): " << format_number(stats.get_coefficient_of_variation() * 100, 2) << "%</li>\n";
        html << "<li>95% confidence interval: [" << format_time(stats.confidence_95_low) << ", " << format_time(stats.confidence_95_high) << "]</li>\n";
        html << "</ul>\n";
    }
    
    html << "</div>\n";
    
    return html.str();
}

/**
 * @brief Format time in human-readable format
 * 
 * @param seconds Time in seconds
 * @return Formatted time string
 */
std::string PerformanceReporter::format_time(double seconds) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << seconds << "s";
    return ss.str();
}

/**
 * @brief Format number with specified precision
 * 
 * @param value Number to format
 * @param precision Number of decimal places
 * @return Formatted number string
 */
std::string PerformanceReporter::format_number(double value, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

} // namespace benchmark
} // namespace pnmatrix

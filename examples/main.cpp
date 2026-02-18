#include "matrix_solver.h"
#include <fstream>  // For file handling

int main() {
    const std::string filename = "matdatat.txt";
    bool printSolution = false;
    int solverChoice = 0;

    // Open a file to write the benchmark results
    std::ofstream benchmarkFile("benchmark_results2.txt");
    if (!benchmarkFile) {
        std::cerr << "Error opening benchmark_results.txt for writing.\n";
        return -1;
    }

    // Ask the user to choose a solver
    std::cout << "Select the solver to use:\n";
    std::cout << "1. Gauss-Seidel\n";
    std::cout << "2. Gaussian Elimination\n";
    std::cout << "Enter choice (1/2): ";
    std::cin >> solverChoice;

    std::cout << "Print solution? (1 = yes, 0 = no): ";
    std::cin >> printSolution;

    // Set matrix sizes for the test
    std::vector<int> matrixSizes = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,20000,30000,40000};
    
    // Loop through each matrix size and perform tests
    for (int matrixSize : matrixSizes) {
        double total_time = 0.0;
        benchmarkFile << "\nRunning tests for matrix size " << matrixSize << " x " << matrixSize << "...\n";
        std::cout << "\nRunning tests for matrix size " << matrixSize << " x " << matrixSize << "...\n";

        // Run 30 iterations for each matrix size
        for (int i = 0; i < 5; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            if (solverChoice == 1) {
                std::cout << "\nRunning Gauss-Seidel for matrix size " << matrixSize << " x " << matrixSize << " (Run " << i + 1 << ")\n";
                pnmatrix::testMatrixGenerationAndSolve<pnmatrix::DenseMatrix>(filename, "Dense", printSolution, matrixSize);
            } else if (solverChoice == 2) {
                std::cout << "\nRunning Gaussian Elimination for matrix size " << matrixSize << " x " << matrixSize << " (Run " << i + 1 << ")\n";
                pnmatrix::testGaussianElimination<pnmatrix::DenseMatrix>(filename, "Dense", printSolution, matrixSize);
            } else {
                std::cerr << "Invalid choice. Exiting...\n";
                return -1;
            }

            auto end = std::chrono::high_resolution_clock::now();
            double run_time = std::chrono::duration<double>(end - start).count();
            total_time += run_time;

            benchmarkFile << "Run " << i + 1 << " time: " << run_time << " seconds\n";
            std::cout << "Run " << i + 1 << " time: " << run_time << " seconds\n";
        }

        double average_time = total_time / 30.0;
        benchmarkFile << "\nAverage time for 30 runs with matrix size " << matrixSize << " x " << matrixSize << ": "
                      << average_time << " seconds\n";

        std::cout << "\nAverage time for 30 runs with matrix size " << matrixSize << " x " << matrixSize << ": "
                  << average_time << " seconds\n";
    }

    std::cout << "\nAll tests completed.\n";
    benchmarkFile << "\nAll tests completed.\n";

    // Close the file after writing
    benchmarkFile.close();

    return 0;
}

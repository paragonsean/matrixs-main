#include "iterative_solvers.h"
#include <exception>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../include/matrix.h"
#include "../include/sparse_matrix_storage.h"

using namespace pnmatrix;

typedef matrix<sparse_matrix_storage<double>> Matrix;

// Helper function to validate matrix dimensions (zero-based indexing)
void validateMatrixDimensions(const Matrix& A, const Matrix& b) {
    if (A.get_column() != A.get_row() || A.get_row() != b.get_row()) {
        throw std::logic_error("Matrix size mismatch: A must be square, and A's rows must match b's rows.");
    }
}

// Helper function for verbose logging (zero-based, no indexing changes needed)
void logVerbose(int iterationCount, double err, const Matrix& x, bool verbose) {
    if (verbose) {
        std::cerr << "Iteration: " << iterationCount << ", Error: " << err << "\n";
        std::cerr << "x: " << x << std::endl;
    }
}

// Jacobi Iterative Solver (zero-based indexing)
int IterativeSolvers::jacobi(const Matrix& A, Matrix& x_old, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;
    Matrix x_new = x_old;

    // Print progress every 20,000 iterations for Jacobi
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations) {
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double rowSum = 0.0;
            for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
                if (j != i) {
                    rowSum += A.get_value(i, j) * x_old.get_value(j, 0);
                }
            }
            x_new.set_value(i, 0, (b.get_value(i, 0) - rowSum) / A.get_value(i, i));
        }
        ++iterationCount;
        err = relError(A, x_new, b);
        logVerbose(iterationCount, err, x_new, verbose);

        // Progress report
        if (iterationCount % reportInterval == 0) {
            std::cout << "Jacobi progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }

        x_old = x_new;
    }

    return iterationCount;
}

// Gauss-Seidel Iterative Solver (zero-based indexing)
int IterativeSolvers::gaussSeidel(const Matrix& A, Matrix& x_old, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;

    // Print progress every 20,000 iterations for Gauss-Seidel
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations) {
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double rowSum = b.get_value(i, 0);
            for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
                if (j != i) {
                    rowSum -= A.get_value(i, j) * x_old.get_value(j, 0);
                }
            }
            x_old.set_value(i, 0, rowSum / A.get_value(i, i));
        }
        ++iterationCount;
        err = relError(A, x_old, b);
        logVerbose(iterationCount, err, x_old, verbose);

        // Progress report
        if (iterationCount % reportInterval == 0) {
            std::cout << "Gauss-Seidel progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }
    }

    return iterationCount;
}

// Successive Over-Relaxation (SOR) Solver (zero-based indexing)
int IterativeSolvers::sor(double omega, const Matrix& A, Matrix& x, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;
    Matrix x_old = x;

    // Print progress every 20,000 iterations for SOR
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations && err < 10.0) {
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double sigma = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma += A.get_value(i, j) * x.get_value(j, 0);
            }
            for (int j = i + 1; j < static_cast<int>(A.get_column()); ++j) {
                sigma += A.get_value(i, j) * x_old.get_value(j, 0);
            }

            double bMinusSigma = b.get_value(i, 0) - sigma;
            double x_old_i = x_old.get_value(i, 0);
            x.set_value(i, 0, x_old_i + omega * (bMinusSigma / A.get_value(i, i) - x_old_i));
        }

        ++iterationCount;
        err = relError(A, x, b);
        logVerbose(iterationCount, err, x, verbose);

        // Progress report
        if (iterationCount % reportInterval == 0) {
            std::cout << "SOR progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }

        x_old = x;
    }

    if (err >= 10.0) {
        std::cerr << "ALERT: SOR diverges using omega = " << omega << ". Try a different relaxation constant.\n";
        std::cerr << "Relative error of SOR solution: " << err << std::endl;
    }

    return iterationCount;
}

// Function to compute the relative error (zero-based indexing)
double IterativeSolvers::relError(const Matrix& A, const Matrix& x, const Matrix& b) {
    double error = 0.0;
    double normB = 0.0;

    for (int i = 0; i < static_cast<int>(b.get_row()); ++i) {
        normB += b.get_value(i, 0) * b.get_value(i, 0);
    }
    normB = std::sqrt(normB);

    if (normB == 0.0) {
        throw std::logic_error("Division by zero: norm of b is zero.");
    }

    Matrix Ax(A.get_row(), 1);
    for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
        double sum = 0.0;
        for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        Ax.set_value(i, 0, sum);
    }

    for (int i = 0; i < static_cast<int>(Ax.get_row()); ++i) {
        double diff = b.get_value(i, 0) - Ax.get_value(i, 0);
        error += diff * diff;
    }
    error = std::sqrt(error);

    return error / normB;
}

/**
 * @file Iterative_solvers.cpp
 * @brief Implementation of iterative solvers for linear systems
 * 
 * This file contains custom implementations of popular iterative methods
 * for solving linear systems Ax = b. These implementations are designed
 * for educational purposes and performance comparison with the library's
 * built-in solvers.
 * 
 * Solvers implemented:
 * - Jacobi iteration
 * - Gauss-Seidel iteration  
 * - Successive Over-Relaxation (SOR)
 * 
 * Features:
 * - Zero-based indexing consistency
 * - Progress reporting for long-running computations
 * - Verbose logging options
 * - Relative error calculation
 * - Convergence detection and divergence warnings
 */

#include "iterative_solvers.h"
#include <exception>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../include/matrix.h"
#include "../include/sparse_matrix_storage.h"

using namespace pnmatrix;

// Type alias for sparse matrix with double precision
typedef matrix<sparse_matrix_storage<double>> Matrix;

/**
 * @brief Validate matrix dimensions for linear system solving
 * 
 * Ensures that the coefficient matrix A is square and compatible
 * with the right-hand side vector b for solving Ax = b.
 * 
 * @param A Coefficient matrix (must be square)
 * @param b Right-hand side vector
 * @throws std::logic_error if dimensions are incompatible
 */
void validateMatrixDimensions(const Matrix& A, const Matrix& b) {
    if (A.get_column() != A.get_row() || A.get_row() != b.get_row()) {
        throw std::logic_error("Matrix size mismatch: A must be square, and A's rows must match b's rows.");
    }
}

/**
 * @brief Verbose logging utility for iterative solver progress
 * 
 * Outputs detailed iteration information including iteration count,
 * current error, and current solution vector when verbose mode is enabled.
 * 
 * @param iterationCount Current iteration number
 * @param err Current relative error
 * @param x Current solution vector
 * @param verbose Whether to output verbose information
 */
void logVerbose(int iterationCount, double err, const Matrix& x, bool verbose) {
    if (verbose) {
        std::cerr << "Iteration: " << iterationCount << ", Error: " << err << "\n";
        std::cerr << "x: " << x << std::endl;
    }
}

/**
 * @brief Jacobi iterative solver for linear systems
 * 
 * Implements the classical Jacobi method for solving Ax = b.
 * The Jacobi method updates all solution components simultaneously
 * using values from the previous iteration.
 * 
 * Algorithm:
 * xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ^(k))) / Aᵢᵢ  (for j ≠ i)
 * 
 * Convergence: Guaranteed for diagonally dominant matrices
 * 
 * @param A Coefficient matrix (must be square, preferably diagonally dominant)
 * @param x_old Initial guess solution vector (modified in place)
 * @param b Right-hand side vector
 * @param errLimit Convergence tolerance (relative error threshold)
 * @param maxIterations Maximum number of iterations to prevent infinite loops
 * @param verbose Enable detailed logging output
 * @return Number of iterations performed
 */
int IterativeSolvers::jacobi(const Matrix& A, Matrix& x_old, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;
    Matrix x_new = x_old;

    // Print progress every 200 iterations for long-running problems
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations) {
        // Jacobi iteration: compute all new values using old iteration values
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double rowSum = 0.0;
            
            // Compute Σ(Aᵢⱼ * xⱼ^(k)) for j ≠ i
            for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
                if (j != i) {
                    rowSum += A.get_value(i, j) * x_old.get_value(j, 0);
                }
            }
            
            // Update formula: xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ^(k))) / Aᵢᵢ
            x_new.set_value(i, 0, (b.get_value(i, 0) - rowSum) / A.get_value(i, i));
        }
        
        ++iterationCount;
        err = relError(A, x_new, b);
        logVerbose(iterationCount, err, x_new, verbose);

        // Progress reporting for long computations
        if (iterationCount % reportInterval == 0) {
            std::cout << "Jacobi progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }

        x_old = x_new;  // Prepare for next iteration
    }

    return iterationCount;
}

/**
 * @brief Gauss-Seidel iterative solver for linear systems
 * 
 * Implements the Gauss-Seidel method, an improvement over Jacobi
 * that uses newly computed values within the same iteration.
 * This typically provides faster convergence.
 * 
 * Algorithm:
 * xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ^(k+1)) - Σ(Aᵢⱼ * xⱼ^(k))) / Aᵢᵢ
 *           (for j < i, use new values; for j > i, use old values)
 * 
 * Convergence: Guaranteed for diagonally dominant or symmetric positive definite matrices
 * 
 * @param A Coefficient matrix (must be square, preferably diagonally dominant)
 * @param x_old Initial guess solution vector (updated in place)
 * @param b Right-hand side vector
 * @param errLimit Convergence tolerance
 * @param maxIterations Maximum number of iterations
 * @param verbose Enable detailed logging output
 * @return Number of iterations performed
 */
int IterativeSolvers::gaussSeidel(const Matrix& A, Matrix& x_old, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;

    // Print progress every 200 iterations
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations) {
        // Gauss-Seidel iteration: use updated values immediately within iteration
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double rowSum = b.get_value(i, 0);
            
            // Subtract contributions from all other variables
            for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
                if (j != i) {
                    rowSum -= A.get_value(i, j) * x_old.get_value(j, 0);
                }
            }
            
            // Update formula: xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ)) / Aᵢᵢ
            x_old.set_value(i, 0, rowSum / A.get_value(i, i));
        }
        
        ++iterationCount;
        err = relError(A, x_old, b);
        logVerbose(iterationCount, err, x_old, verbose);

        // Progress reporting
        if (iterationCount % reportInterval == 0) {
            std::cout << "Gauss-Seidel progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }
    }

    return iterationCount;
}

/**
 * @brief Successive Over-Relaxation (SOR) solver for linear systems
 * 
 * Implements the SOR method, an acceleration of Gauss-Seidel that uses
 * a relaxation parameter ω to improve convergence speed.
 * 
 * Algorithm:
 * xᵢ^(k+1) = xᵢ^(k) + ω * (xᵢ^(GS) - xᵢ^(k))
 * where xᵢ^(GS) is the Gauss-Seidel update
 * 
 * Relaxation parameter:
 * - ω = 1: Equivalent to Gauss-Seidel
 * - 0 < ω < 2: Required for convergence
 * - ω > 1: Over-relaxation (typically faster convergence)
 * - ω < 1: Under-relaxation (more stable but slower)
 * 
 * @param omega Relaxation parameter (0 < ω < 2 for convergence)
 * @param A Coefficient matrix
 * @param x Initial guess solution vector (updated in place)
 * @param b Right-hand side vector
 * @param errLimit Convergence tolerance
 * @param maxIterations Maximum number of iterations
 * @param verbose Enable detailed logging output
 * @return Number of iterations performed
 */
int IterativeSolvers::sor(double omega, const Matrix& A, Matrix& x, const Matrix& b, double errLimit, int maxIterations, bool verbose) {
    validateMatrixDimensions(A, b);

    int iterationCount = 0;
    double err = 1.0;
    Matrix x_old = x;

    // Print progress every 200 iterations
    int reportInterval = 200;

    while (err > errLimit && iterationCount < maxIterations && err < 10.0) {
        // SOR iteration: weighted update between old and Gauss-Seidel values
        for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
            double sigma = 0.0;
            
            // Sum contributions from already updated variables (j < i)
            for (int j = 0; j < i; ++j) {
                sigma += A.get_value(i, j) * x.get_value(j, 0);
            }
            
            // Sum contributions from not-yet-updated variables (j > i)
            for (int j = i + 1; j < static_cast<int>(A.get_column()); ++j) {
                sigma += A.get_value(i, j) * x_old.get_value(j, 0);
            }

            // Compute Gauss-Seidel update
            double bMinusSigma = b.get_value(i, 0) - sigma;
            double x_old_i = x_old.get_value(i, 0);
            double x_gs = bMinusSigma / A.get_value(i, i);
            
            // Apply relaxation: xᵢ^(k+1) = xᵢ^(k) + ω * (xᵢ^(GS) - xᵢ^(k))
            x.set_value(i, 0, x_old_i + omega * (x_gs - x_old_i));
        }

        ++iterationCount;
        err = relError(A, x, b);
        logVerbose(iterationCount, err, x, verbose);

        // Progress reporting
        if (iterationCount % reportInterval == 0) {
            std::cout << "SOR progress: " << iterationCount << " iterations, current error: " << err << std::endl;
        }

        x_old = x;  // Store current iteration for next SOR update
    }

    // Divergence detection and warning
    if (err >= 10.0) {
        std::cerr << "ALERT: SOR diverges using omega = " << omega << ". Try a different relaxation constant.\n";
        std::cerr << "Relative error of SOR solution: " << err << std::endl;
    }

    return iterationCount;
}

/**
 * @brief Compute relative error for iterative solver convergence
 * 
 * Calculates the relative residual error: ||b - Ax|| / ||b||
 * This measures how well the current solution satisfies the original equation.
 * 
 * Mathematical definition:
 * relative_error = ||b - Ax||₂ / ||b||₂
 * 
 * where ||·||₂ is the Euclidean (L2) norm.
 * 
 * @param A Coefficient matrix
 * @param x Current solution vector
 * @param b Right-hand side vector
 * @return Relative error (0.0 = perfect solution, larger values = worse solution)
 * @throws std::logic_error if norm of b is zero (division by zero)
 */
double IterativeSolvers::relError(const Matrix& A, const Matrix& x, const Matrix& b) {
    double error = 0.0;
    double normB = 0.0;

    // Compute ||b||₂ (Euclidean norm of b)
    for (int i = 0; i < static_cast<int>(b.get_row()); ++i) {
        normB += b.get_value(i, 0) * b.get_value(i, 0);
    }
    normB = std::sqrt(normB);

    if (normB == 0.0) {
        throw std::logic_error("Division by zero: norm of b is zero.");
    }

    // Compute Ax (matrix-vector product)
    Matrix Ax(A.get_row(), 1);
    for (int i = 0; i < static_cast<int>(A.get_row()); ++i) {
        double sum = 0.0;
        for (int j = 0; j < static_cast<int>(A.get_column()); ++j) {
            sum += A.get_value(i, j) * x.get_value(j, 0);
        }
        Ax.set_value(i, 0, sum);
    }

    // Compute ||b - Ax||₂ (Euclidean norm of residual)
    for (int i = 0; i < static_cast<int>(Ax.get_row()); ++i) {
        double diff = b.get_value(i, 0) - Ax.get_value(i, 0);
        error += diff * diff;
    }
    error = std::sqrt(error);

    return error / normB;  // Relative error
}

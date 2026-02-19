/**
 * @file matrix_solvers.h
 * @brief Template-based direct solvers for linear systems
 * 
 * This header provides implementations of direct methods for solving
 * linear systems Ax = b using template metaprogramming to work with
 * different matrix storage types (dense and sparse).
 * 
 * Solvers implemented:
 * - Gaussian Elimination with partial pivoting
 * - LU Decomposition with forward/back substitution
 * 
 * Features:
 * - Thread-safe printing utilities
 * - Numerical stability checks
 * - Relative error computation
 * - Exception handling for singular matrices
 * - Template-based design for storage abstraction
 */

#ifndef MATRIXSOLVERS_H_
#define MATRIXSOLVERS_H_

#include "../include/matrix.h"
#include <cmath>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace pnmatrix {

/**
 * @brief Thread-safe printing utility for multi-threaded solver operations
 * 
 * Provides mutex-protected console output to prevent interleaved messages
 * when running solvers in parallel or multi-threaded environments.
 */
std::mutex coutMutex;

/**
 * @brief Thread-safe console printing function
 * 
 * Uses a mutex to ensure that output from different threads doesn't
 * get interleaved, providing clean, readable console output.
 * 
 * @param message Message to print to console
 */
void safePrint(const std::string& message) {
    std::lock_guard<std::mutex> guard(coutMutex);
    std::cout << message;
}

/**
 * @brief Template-based direct solver class for linear systems
 * 
 * This class provides direct methods for solving linear systems Ax = b
 * using template metaprogramming to work with any matrix storage type
 * that provides the required interface (get_value, set_value, etc.).
 * 
 * Template parameter allows the same implementation to work with:
 * - Dense matrix storage
 * - Sparse matrix storage
 * - Custom storage implementations
 * 
 * @tparam MatrixType Matrix storage type (must provide standard matrix interface)
 */
template <typename MatrixType>
class matrixSolvers {
public:
    /**
     * @brief Find pivot element for partial pivoting in Gaussian elimination
     * 
     * Implements partial pivoting strategy by finding the element with
     * maximum absolute value in the current column below the diagonal.
     * This improves numerical stability and prevents division by small numbers.
     * 
     * @param A Coefficient matrix (modified during elimination)
     * @param column Current column index for pivot selection
     * @return Row index of the best pivot element
     * @throws std::logic_error if matrix is singular (no suitable pivot found)
     */
    int findPivot(const MatrixType& A, int column);

    /**
     * @brief Back substitution solver for upper triangular systems
     * 
     * Solves Ux = y where U is an upper triangular matrix using
     * back substitution (solving from bottom to top).
     * 
     * Algorithm: xᵢ = (yᵢ - Σ(Uᵢⱼ * xⱼ)) / Uᵢᵢ  for j > i
     * 
     * @param U Upper triangular matrix
     * @param y Right-hand side vector
     * @param x Output solution vector
     * @throws std::logic_error if diagonal element is zero (singular matrix)
     */
    void backsolve(const MatrixType& U, const MatrixType& y, MatrixType& x);
    
    /**
     * @brief Forward substitution solver for lower triangular systems
     * 
     * Solves Ly = b where L is a lower triangular matrix using
     * forward substitution (solving from top to bottom).
     * 
     * Algorithm: yᵢ = (bᵢ - Σ(Lᵢⱼ * yⱼ)) / Lᵢᵢ  for j < i
     * 
     * @param L Lower triangular matrix
     * @param y Output intermediate solution vector
     * @param b Right-hand side vector
     * @throws std::logic_error if diagonal element is zero (singular matrix)
     */
    void forwardSolve(MatrixType& L, MatrixType& y, const MatrixType& b);

    /**
     * @brief Gaussian elimination with partial pivoting
     * 
     * Implements the classical Gaussian elimination algorithm with
     * partial pivoting for numerical stability. Transforms the
     * augmented matrix [A|b] into upper triangular form, then uses
     * back substitution to find the solution.
     * 
     * Algorithm steps:
     * 1. For each column i, find pivot row with maximum |A[row,i]|
     * 2. Swap current row with pivot row
     * 3. Eliminate elements below pivot using row operations
     * 4. Solve resulting upper triangular system
     * 
     * @param A Coefficient matrix (modified to upper triangular form)
     * @param x Output solution vector
     * @param b Right-hand side vector (modified during elimination)
     * @throws std::logic_error if matrix is singular
     */
    void gaussianElimination(MatrixType& A, MatrixType& x, const MatrixType& b);
    
    /**
     * @brief LU decomposition with forward/back substitution
     * 
     * Decomposes matrix A into L (lower triangular) and U (upper triangular)
     * matrices such that A = LU. Then solves Ly = b (forward substitution)
     * and Ux = y (back substitution) to find the solution.
     * 
     * This method is more efficient than Gaussian elimination when
     * solving multiple systems with the same coefficient matrix.
     * 
     * Algorithm (Doolittle decomposition):
     * - L has unit diagonal, U has non-zero diagonal
     * - U[i,k] = A[i,k] - Σ(L[i,j] * U[j,k]) for j < i
     * - L[k,i] = (A[k,i] - Σ(L[k,j] * U[j,i])) / U[i,i] for j < i
     * 
     * @param A Coefficient matrix (not modified)
     * @param x Output solution vector
     * @param b Right-hand side vector
     * @throws std::logic_error if matrix is singular
     */
    void LUdecomposition(MatrixType& A, MatrixType& x, const MatrixType& b);

private:
    /**
     * @brief Compute relative error for solution validation
     * 
     * Calculates the relative residual error: ||Ax - b|| / ||b||
     * This measures how accurately the computed solution satisfies
     * the original linear system.
     * 
     * Mathematical definition:
     * relative_error = ||Ax - b||₂ / ||b||₂
     * 
     * @param A Coefficient matrix
     * @param x Computed solution vector
     * @param b Original right-hand side vector
     * @return Relative error (smaller values indicate better solutions)
     */
    double computeRelativeError(const MatrixType& A, const MatrixType& x, const MatrixType& b);
};

// Implementation of matrixSolvers methods

/**
 * @brief Implementation of relative error computation
 * 
 * Computes the Euclidean norm of the residual vector (Ax - b) and
 * normalizes it by the Euclidean norm of b to get a relative error measure.
 */
template <typename MatrixType>
double matrixSolvers<MatrixType>::computeRelativeError(const MatrixType& A, const MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    double numerator = 0.0, denominator = 0.0;

    // Compute ||Ax - b||₂ (numerator)
    for (size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (size_t j = 0; j < A.get_column(); ++j) {
            Ax_i += A.get_value(i, j) * x.get_value(j, 0);
        }
        double residual = Ax_i - b.get_value(i, 0);
        numerator += residual * residual;
    }

    // Compute ||b||₂ (denominator)
    for (size_t i = 0; i < n; ++i) {
        denominator += b.get_value(i, 0) * b.get_value(i, 0);
    }

    // Return relative error with small epsilon to prevent division by zero
    return std::sqrt(numerator) / (std::sqrt(denominator) + 1e-6);
}

/**
 * @brief Implementation of pivot finding with partial pivoting
 * 
 * Searches for the element with maximum absolute value in the current
 * column from the diagonal element downward. This is the standard
 * partial pivoting strategy for numerical stability.
 */
template <typename MatrixType>
int matrixSolvers<MatrixType>::findPivot(const MatrixType& A, int column) {
    int pivotRow = column;
    double maxVal = 0.0;

    // Find maximum absolute value in column 'column' from row 'column' downward
    for (size_t i = column; i < A.get_row(); ++i) {
        double val = std::abs(A.get_value(i, column));
        if (val > maxVal) {
            maxVal = val;
            pivotRow = i;
        }
    }

    // Check for singular matrix (pivot too small)
    if (maxVal < 1e-6) {
        throw std::logic_error("Matrix is singular.");
    }

    return pivotRow;
}

/**
 * @brief Implementation of back substitution
 * 
 * Solves upper triangular system Ux = y by working from the
 * last equation upward, substituting already-computed values.
 */
template <typename MatrixType>
void matrixSolvers<MatrixType>::backsolve(const MatrixType& U, const MatrixType& y, MatrixType& x) {
    size_t n = U.get_row();
    x = MatrixType(n, 1);

    // Solve from bottom to top: x[n-1], x[n-2], ..., x[0]
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = 0.0;
        
        // Compute Σ(U[i,j] * x[j]) for j > i
        for (size_t j = i + 1; j < n; ++j) {
            sum += U.get_value(i, j) * x.get_value(j, 0);
        }
        
        double diag = U.get_value(i, i);
        if (std::abs(diag) < 1e-6) {
            throw std::logic_error("Division by zero in backsolve.");
        }
        
        // x[i] = (y[i] - sum) / U[i,i]
        x.set_value(i, 0, (y.get_value(i, 0) - sum) / diag);
    }
}

/**
 * @brief Implementation of forward substitution
 * 
 * Solves lower triangular system Ly = b by working from the
 * first equation downward, substituting already-computed values.
 */
template <typename MatrixType>
void matrixSolvers<MatrixType>::forwardSolve(MatrixType& L, MatrixType& y, const MatrixType& b) {
    size_t n = L.get_row();
    y = MatrixType(n, 1);

    // Solve from top to bottom: y[0], y[1], ..., y[n-1]
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        
        // Compute Σ(L[i,j] * y[j]) for j < i
        for (size_t j = 0; j < i; ++j) {
            sum += L.get_value(i, j) * y.get_value(j, 0);
        }
        
        double diag = L.get_value(i, i);
        if (std::abs(diag) < 1e-6) {
            throw std::logic_error("Division by zero in forwardSolve.");
        }
        
        // y[i] = (b[i] - sum) / L[i,i]
        y.set_value(i, 0, (b.get_value(i, 0) - sum) / diag);
    }
}

/**
 * @brief Implementation of Gaussian elimination with partial pivoting
 * 
 * Transforms the augmented matrix [A|b] into upper triangular form
 * through row operations, then solves using back substitution.
 */
template <typename MatrixType>
void matrixSolvers<MatrixType>::gaussianElimination(MatrixType& A, MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    x = MatrixType(n, 1);

    // Forward elimination phase
    for (size_t i = 0; i < n; ++i) {
        // Partial pivoting: find and swap with best pivot row
        int pivot = findPivot(A, i);
        A.element_row_transform_swap(i, pivot);
        x.element_row_transform_swap(i, pivot);

        // Eliminate elements below pivot in column i
        for (size_t j = i + 1; j < n; ++j) {
            double factor = A.get_value(j, i) / A.get_value(i, i);
            
            // Row operation: Row[j] = Row[j] - factor * Row[i]
            for (size_t k = i; k < n; ++k) {
                A.add_value(j, k, -factor * A.get_value(i, k));
            }
            x.add_value(j, 0, -factor * x.get_value(i, 0));
        }
    }

    // Back substitution phase
    backsolve(A, x, x);
}

/**
 * @brief Implementation of LU decomposition (Doolittle method)
 * 
 * Decomposes A into L (unit lower triangular) and U (upper triangular)
 * matrices, then solves using forward and back substitution.
 */
template <typename MatrixType>
void matrixSolvers<MatrixType>::LUdecomposition(MatrixType& A, MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    MatrixType L(n, n), U(n, n);

    // Doolittle decomposition: L has unit diagonal, U has variable diagonal
    for (size_t i = 0; i < n; ++i) {
        // Compute U[i,k] for k >= i
        for (size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += L.get_value(i, j) * U.get_value(j, k);
            }
            U.set_value(i, k, A.get_value(i, k) - sum);
        }

        // Compute L[k,i] for k >= i
        for (size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += L.get_value(k, j) * U.get_value(j, i);
            }
            L.set_value(k, i, (A.get_value(k, i) - sum) / U.get_value(i, i));
        }
    }

    // Solve Ly = b (forward substitution)
    MatrixType y(n, 1);
    forwardSolve(L, y, b);
    
    // Solve Ux = y (back substitution)
    backsolve(U, y, x);
}

} // namespace pnmatrix

#endif /* MATRIXSOLVERS_H_ */

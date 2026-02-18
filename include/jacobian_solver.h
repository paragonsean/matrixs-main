#pragma once

#include "type.h"
#include "value_compare.h"
#include <utility>
#include <cassert>
#include <cmath>
#include <iostream>

namespace pnmatrix {
/**
 * @brief Jacobi iterative solver for linear systems Ax = b
 * 
 * The Jacobi method is an iterative algorithm for solving systems of linear equations.
 * It's particularly useful for diagonally dominant systems and as a foundation for
 * more advanced iterative methods.
 * 
 * Algorithm:
 * 1. Start with initial guess x₀ (usually zero vector)
 * 2. For each iteration, compute new solution using:
 *    xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ^(k))) / Aᵢᵢ  (for j ≠ i)
 * 3. Check convergence: ||Ax^(k+1) - b|| < tolerance
 * 4. Repeat until convergence or max iterations
 * 
 * Convergence condition: System should be diagonally dominant or symmetric positive definite
 */
class jacobian {
private:
  double rm_;  // Residual tolerance for convergence checking

public:
  struct option {
    double rm = 1e-6;  // Default tolerance: 1e-6
  };

  jacobian(option op):rm_(op.rm) {}

  /**
   * @brief Solve linear system Ax = b using Jacobi iteration
   * @tparam MatrixType Matrix container type (dense or sparse)
   * @param coeff Coefficient matrix A
   * @param b Right-hand side vector
   * @return Solution vector x
   * 
   * Time complexity: O(n² * iterations) for dense matrices
   * Space complexity: O(n) for solution vectors
   */
  template<class MatrixType>
  MatrixType solve(MatrixType& coeff, MatrixType& b) {
    // Validate input dimensions: A must be square, b must be column vector
    assert(coeff.get_column() == b.get_row() && b.get_column() == 1);
    
    size_type x_count = coeff.get_column();
    // Initialize solution vectors (x_prev for previous iteration, x_next for current)
    MatrixType x_prev(x_count, 1);  // Previous iteration solution
    MatrixType x_next(x_count, 1);  // Current iteration solution
    size_t times = 1;  // Iteration counter

    // Main iteration loop
    while (true) {
      // Jacobi iteration: compute each component of new solution
      for (size_type row = 0; row < coeff.get_row(); ++row) {
        double sum = 0.0;
        
        // Compute Σ(Aᵢⱼ * xⱼ^(k)) for all j ≠ i
        // This is the contribution from all other variables
        for (size_type colu = 0; colu < coeff.get_column(); ++colu) {
          if (colu != row) {  // Skip diagonal element
            sum += coeff.get_value(row, colu) * x_prev.get_value(colu, 0);
          }
        }

        // Extract diagonal element Aᵢᵢ and right-hand side bᵢ
        double diag = coeff.get_value(row, row);
        double rhs = b.get_value(row, 0);
        
        // Jacobi formula: xᵢ^(k+1) = (bᵢ - Σ(Aᵢⱼ * xⱼ^(k))) / Aᵢᵢ
        double result = (rhs - sum) / diag;

        // Handle singular matrix (zero diagonal element)
        if (value_equal(diag, 0.0) == true) {
          x_next.set_value(row, 0, 0.0);  // Set to zero if diagonal is zero
        } else {
          x_next.set_value(row, 0, result);  // Store computed value
        }
      }

      // Convergence check: compute residual ||Ax - b||
      MatrixType tmp = coeff * x_next;  // Compute Ax^(k+1)
      double max_err = max_error(tmp, b);  // Compute ||Ax - b||∞
      
      if (max_err <= rm_) {
        break;  // Converged - exit loop
      } else {
        std::swap(x_prev, x_next);  // Prepare for next iteration
        ++times;  // Increment iteration counter
      }
    }

    return x_next;  // Return converged solution
  }

private:
  /**
   * @brief Compute maximum absolute error between two matrices
   * @tparam MatrixType Matrix container type
   * @param m1 First matrix
   * @param m2 Second matrix
   * @return Maximum absolute difference ||m1 - m2||∞
   * 
   * This computes the infinity norm of the difference matrix,
   * which is the maximum absolute value of any element.
   */
  template<class MatrixType>
  double max_error(const MatrixType& m1, const MatrixType& m2) {
    // Validate that matrices have same dimensions
    assert(m1.get_row() == m2.get_row() && m1.get_column() == m2.get_column());
    
    double max_err = 0.0;
    
    // Find maximum absolute difference between corresponding elements
    for (size_type row = 0; row < m1.get_row(); ++row) {
      for (size_type colu = 0; colu < m1.get_column(); ++colu) {
        double error = m1.get_value(row, colu) - m2.get_value(row, colu);
        error = std::abs(error);  // Take absolute value
        if (error > max_err) {
          max_err = error;  // Update maximum error found
        }
      }
    }
    return max_err;
  }
};
}

#pragma once

#include "type.h"
#include "value_compare.h"
#include <cassert>
#include <cmath>
#include <iostream>

namespace pnmatrix {

class gaussian_elimination {
private:
    double rm_;  // Convergence tolerance (not used in Gaussian Elimination, but can be useful for iterative methods)

public:
    struct option {
        double rm = 1e-6;  // Not necessary for Gaussian Elimination, but kept for consistency
    };

    gaussian_elimination(option op): rm_(op.rm) {}

    // Function to perform Gaussian Elimination and solve the system Ax = b
    template<class MatrixType>
    MatrixType solve(MatrixType& coeff, MatrixType& b) {
        assert(coeff.get_row() == coeff.get_column() && coeff.get_column() == b.get_row());
        
        size_type n = coeff.get_row();
        MatrixType augmented(n, n + 1);  // Augmented matrix [A | b]
        
        // Construct the augmented matrix by combining A and b
        for (size_type i = 0; i < n; ++i) {
            for (size_type j = 0; j < n; ++j) {
                augmented.set_value(i, j, coeff.get_value(i, j));
            }
            augmented.set_value(i, n, b.get_value(i, 0));  // Last column is the vector b
        }

        // Perform Gaussian Elimination
        size_type iteration_count = 0; // Counter for iterations
        for (size_type i = 0; i < n; ++i) {
            // Pivoting: Find the row with the largest value in the current column
            size_type max_row = i;
            for (size_type k = i + 1; k < n; ++k) {
                if (std::abs(augmented.get_value(k, i)) > std::abs(augmented.get_value(max_row, i))) {
                    max_row = k;
                }
            }

            // Swap the rows if necessary
            if (i != max_row) {
                for (size_type j = 0; j < n + 1; ++j) {
                    double temp = augmented.get_value(i, j);
                    augmented.set_value(i, j, augmented.get_value(max_row, j));
                    augmented.set_value(max_row, j, temp);
                }
            }

            // Eliminate the elements below the pivot
            for (size_type j = i + 1; j < n; ++j) {
                double factor = augmented.get_value(j, i) / augmented.get_value(i, i);
                for (size_type k = i; k < n + 1; ++k) {
                    augmented.set_value(j, k, augmented.get_value(j, k) - factor * augmented.get_value(i, k));
                }
            }

            // Print status every 100,000 iterations
            iteration_count++;
            if (iteration_count % 100000 == 0) {
                std::cout << "Processed " << iteration_count << " iterations...\n";
            }
        }

        // Back Substitution to solve for x
        MatrixType x(n, 1);
        for (size_type i = n - 1; i >= 0; --i) {
            double sum = augmented.get_value(i, n);  // Last column is b
            for (size_type j = i + 1; j < n; ++j) {
                sum -= augmented.get_value(i, j) * x.get_value(j, 0);
            }
            x.set_value(i, 0, sum / augmented.get_value(i, i));
        }

        return x;  // Return the solution vector
    }
};

} // namespace pnmatrix

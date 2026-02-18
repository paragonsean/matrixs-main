#pragma once
#include "type.h"
#include "value_compare.h"
#include "qr_decomposition.h"
#include <utility>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace pnmatrix {
class gmres {
private:
    double residual_tolerance_;
    size_type restart_size_;

public:
    struct option {
        double rm = 1e-6;
        size_type m = 30; // Changed from double to size_type
    };

    // Utility function to print matrices
    template <typename MatrixType>
    void print_matrix(const MatrixType& m) const {
        for (auto row = m.begin(); row != m.end(); ++row) {
            for (auto col = row.begin(); col != row.end(); ++col) {
                std::cout << "(" << col.row_index() << ", " << col.column_index() << ") " << *col << " ";
            }
            std::cout << "\n";
        }
    }

    gmres(option op) : residual_tolerance_(op.rm), restart_size_(op.m) {}

    template <class MatrixType>
    MatrixType solve(MatrixType& A, MatrixType& b) {
        // Ensure m does not exceed the size of A
        size_type n = A.get_row();
        if (restart_size_ > n) {
            std::cerr << "Warning: Restart parameter m (" << restart_size_
                      << ") exceeds matrix size (" << n << "). Setting m to " << n << ".\n";
            restart_size_ = n;
        }

        // Initialize x0 with zeros
        MatrixType x0(b.get_row(), b.get_column());
        for (size_type i = 0; i < x0.get_row(); ++i) {
            for (size_type j = 0; j < x0.get_column(); ++j) {
                x0.set_value(i, j, 0.0);
            }
        }
        return solveinner(A, b, x0, restart_size_);
    }

private:
    template <class MatrixType>
    MatrixType solveinner(MatrixType& A, MatrixType& b, MatrixType& x0, size_type restart_m) {
        using value_type = typename MatrixType::value_type;
        MatrixType result(x0.get_row(), x0.get_column());

        while (true) {
            // Compute initial residual r0 = b - A*x0
            MatrixType r0 = b - (A * x0);
            value_type beta = r0.get_vector_second_norm();

            // Debugging: Print beta and residual
            std::cout << "Initial residual norm (beta): " << beta << "\n";

            // Check if the initial residual is already within tolerance
            if (beta <= residual_tolerance_) {
                std::cout << "Converged with initial residual.\n";
                return x0;
            }

            // Vm will hold the Krylov basis vectors
            MatrixType Vm(A.get_row(), 1);
            MatrixType v1 = r0 / beta; // v1 = r0 / ||r0||
            Vm.set_column(0, v1);      // set first column (index 0)

            // Debugging: Print Vm
            std::cout << "Vm after first vector:\n";
            print_matrix(Vm);

            // H will eventually be (restart_m +1) x restart_m
            MatrixType H(restart_m + 1, restart_m);

            for (size_type iter = 0; iter < restart_m; ++iter) {
                // Resize H to (iter +2) x (iter +1)
                H.resize(iter + 2, iter + 1);
                std::cout << "Resized H to (" << H.get_row() << "x" << H.get_column() << ")\n";

                // Extract the iter-th column of Vm (0-based)
                MatrixType V_col = Vm.get_nth_column(iter);
                MatrixType W_vector = A * V_col;

                // Transpose of W_vector
                MatrixType W_transpose = tr(W_vector);

                // Orthogonalize W_vector against existing basis vectors in Vm
                for (size_type i = 0; i <= iter; ++i) {
                    // Extract i-th column of Vm as a vector
                    // Assuming get_sub_matrix takes (start_row, end_row, start_col, num_cols)
                    MatrixType V_col_i = Vm.get_sub_matrix(0, Vm.get_row(), i, 1);
                    value_type dot_product = W_transpose.get_vector_inner_product(V_col_i);
                    H.set_value(i, iter, dot_product);
                    W_vector = W_vector - (V_col_i * dot_product);
                }

                value_type norm_w = W_vector.get_vector_second_norm();
                H.set_value(iter + 1, iter, norm_w);

                // Debugging: Print H and norm_w
                std::cout << "Iteration " << iter + 1 << ":\n";
                print_matrix(H);
                std::cout << "Norm of W_vector: " << norm_w << "\n";

                // Check if the new vector is near zero, indicating breakdown
                if (value_equal(norm_w, value_type(0))) {
                    std::cerr << "Breakdown detected: norm_w is zero.\n";
                    return result; // Or handle as needed
                }

                // Perform QR decomposition of H
                // Initialize QR_factorization directly with QR<MatrixType>(H)
                std::pair<MatrixType, MatrixType> QR_factorization = QR<MatrixType>(H);
                // Or using auto:
                // auto QR_factorization = QR<MatrixType>(H);
                std::cout << "QR Decomposition completed.\n";

                // Q_first_column is the first column of Q (0-based index 0)
                MatrixType Q_first_column = QR_factorization.first.get_nth_column(0);
                std::cout << "Q_first_column before deletion:\n";
                print_matrix(Q_first_column);

                // Delete the last row of Q_first_column
                if (Q_first_column.get_row() == 0 || Q_first_column.get_column() == 0) {
                    throw std::invalid_argument("Q_first_column has invalid dimensions after deletion.");
                }
                Q_first_column.delete_row(Q_first_column.get_row() - 1);
                std::cout << "Q_first_column after deletion:\n";
                print_matrix(Q_first_column);

                MatrixType R_matrix = std::move(QR_factorization.second);
                R_matrix.delete_row(R_matrix.get_row() - 1);
                std::cout << "R_matrix after deletion:\n";
                print_matrix(R_matrix);

                // Invert R_matrix
                MatrixType M_inverse(R_matrix.get_row(), R_matrix.get_column());
                bool inversion_success = R_matrix.inverse_with_ert(M_inverse);
                if (!inversion_success) {
                    std::cerr << "Error: Matrix inversion failed during GMRES iterations.\n";
                    throw std::invalid_argument("Matrix inversion failed.");
                }
                std::cout << "R_matrix inverted successfully.\n";

                // Compute residual ratio
                value_type residual_ratio = QR_factorization.first.get_value(iter + 1, 0) * (beta / b.get_vector_second_norm());
                std::cout << "Residual ratio: " << residual_ratio << "\n";

                // Check convergence
                if (std::abs(residual_ratio) <= residual_tolerance_) {
                    result = x0 + (Vm * M_inverse * Q_first_column * beta);
                    std::cout << "Converged at iteration " << iter + 1 << ".\n";
                    return result;
                }

                // If reached end of restart steps without convergence
                if (iter == restart_m - 1) {
                    result = x0 + (Vm * M_inverse * Q_first_column * beta);
                }

                // Ensure norm_w is not zero
                if (value_equal(value_type(0), norm_w)) {
                    std::cerr << "Error: norm_w is zero after operations.\n";
                    throw std::invalid_argument("norm_w is zero.");
                }

                // Resize Vm to add next vector
                Vm.resize(A.get_row(), iter + 2);
                Vm.set_column(iter + 1, W_vector / norm_w);
                std::cout << "Vm after adding new vector:\n";
                print_matrix(Vm);
            }

            // Update x0 to the current best solution and restart
            x0 = std::move(result);
        }
    };
}; // namespace pnmatrix
}
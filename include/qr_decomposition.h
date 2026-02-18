#pragma once
#include "type.h"
#include "value_compare.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace pnmatrix {

template<class MatrixType>
MatrixType get_householder_matrix(const MatrixType& vector) {
    assert(vector.get_column() == 1);
    bool zero_vec = true;
    using value_type = typename MatrixType::value_type;

    // Check if the vector is zero
    for (size_type i = 0; i < vector.get_row(); ++i) {
        if (!value_equal(vector.get_value(i, 0), value_type(0))) {
            zero_vec = false;
            break;
        }
    }
    assert(!zero_vec && "Vector is zero; Householder transformation undefined.");

    // Compute the norm of the vector
    value_type norm = vector.get_vector_second_norm();

    // Create a temporary vector for Householder
    MatrixType tmp(vector.get_row(), 1);
    tmp.set_value(0, 0, norm);

    // If the vector is already aligned with the first basis vector
    if (tmp == vector) {
        MatrixType w(vector.get_row(), 1);
        if (vector.get_row() > 1) {
            w.set_value(1, 0, 1.0); // Set the second element to 1
        }
        MatrixType H = MatrixType::get_identity_matrix(vector.get_row());
        return H - w * tr(w) * 2.0;
    }

    // Compute Householder vector
    MatrixType x_au = vector - tmp;
    value_type x_au_norm = x_au.get_vector_second_norm();
    if (value_equal(x_au_norm, value_type(0))) {
        throw std::invalid_argument("Householder vector norm is zero.");
    }
    MatrixType w = x_au / x_au_norm;
    MatrixType H = MatrixType::get_identity_matrix(vector.get_row());
    H = H - w * tr(w) * 2.0;
    return H;
}

template<class MatrixType>
std::pair<MatrixType, MatrixType> QR(const MatrixType& matrix) {
    if (matrix.get_row() <= matrix.get_column()) {
        fprintf(stderr, "QR decomposition requires m > n (rows > columns).\n");
        std::abort();
    }

    MatrixType R(matrix);
    MatrixType Q = MatrixType::get_identity_matrix(matrix.get_row());

    for (size_type i = 0; i < matrix.get_column(); ++i) { // 0-based indexing
        // Extract the i-th column below the diagonal
        MatrixType tmp = R.get_sub_matrix(i, matrix.get_row() - i, i, 1); // (start_row, rows, start_col, cols)
        
        // Compute Householder matrix
        MatrixType H = get_householder_matrix(tmp);
        
        // Expand H to full size
        MatrixType H_full = MatrixType::get_identity_matrix(matrix.get_row());
        H_full.set_value_from_matrix(i, i, H);
        
        // Update Q
        Q = H_full * Q;
        
        // Update R
        MatrixType A_i = R.get_sub_matrix(i, matrix.get_row() - i, i, matrix.get_column() - i);
        MatrixType hai = H * A_i;
        R.set_value_from_matrix(i, i, hai);
    }

    return { Q, R };
}

} // namespace pnmatrix

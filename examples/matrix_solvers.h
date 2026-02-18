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

// Safe printing using mutex
std::mutex coutMutex;
void safePrint(const std::string& message) {
    std::lock_guard<std::mutex> guard(coutMutex);
    std::cout << message;
}

template <typename MatrixType>
class matrixSolvers {
public:
    int findPivot(const MatrixType& A, int column);

    void backsolve(const MatrixType& U, const MatrixType& y, MatrixType& x);
    void forwardSolve(MatrixType& L, MatrixType& y, const MatrixType& b);

    void gaussianElimination(MatrixType& A, MatrixType& x, const MatrixType& b);
    void LUdecomposition(MatrixType& A, MatrixType& x, const MatrixType& b);

private:
    double computeRelativeError(const MatrixType& A, const MatrixType& x, const MatrixType& b);
};

// Implementation of matrixSolvers methods

template <typename MatrixType>
double matrixSolvers<MatrixType>::computeRelativeError(const MatrixType& A, const MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    double numerator = 0.0, denominator = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (size_t j = 0; j < A.get_column(); ++j) {
            Ax_i += A.get_value(i, j) * x.get_value(j, 0);
        }
        double residual = Ax_i - b.get_value(i, 0);
        numerator += residual * residual;
        denominator += b.get_value(i, 0) * b.get_value(i, 0);
    }

    return std::sqrt(numerator) / (std::sqrt(denominator) + 1e-6);
}

template <typename MatrixType>
int matrixSolvers<MatrixType>::findPivot(const MatrixType& A, int column) {
    int pivotRow = column;
    double maxVal = 0.0;

    for (size_t i = column; i < A.get_row(); ++i) {
        double val = std::abs(A.get_value(i, column));
        if (val > maxVal) {
            maxVal = val;
            pivotRow = i;
        }
    }

    if (maxVal < 1e-6) {
        throw std::logic_error("Matrix is singular.");
    }

    return pivotRow;
}

template <typename MatrixType>
void matrixSolvers<MatrixType>::backsolve(const MatrixType& U, const MatrixType& y, MatrixType& x) {
    size_t n = U.get_row();
    x = MatrixType(n, 1);

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += U.get_value(i, j) * x.get_value(j, 0);
        }
        double diag = U.get_value(i, i);
        if (std::abs(diag) < 1e-6) {
            throw std::logic_error("Division by zero in backsolve.");
        }
        x.set_value(i, 0, (y.get_value(i, 0) - sum) / diag);
    }
}

template <typename MatrixType>
void matrixSolvers<MatrixType>::forwardSolve(MatrixType& L, MatrixType& y, const MatrixType& b) {
    size_t n = L.get_row();
    y = MatrixType(n, 1);

    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L.get_value(i, j) * y.get_value(j, 0);
        }
        double diag = L.get_value(i, i);
        if (std::abs(diag) < 1e-6) {
            throw std::logic_error("Division by zero in forwardSolve.");
        }
        y.set_value(i, 0, (b.get_value(i, 0) - sum) / diag);
    }
}

template <typename MatrixType>
void matrixSolvers<MatrixType>::gaussianElimination(MatrixType& A, MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    x = MatrixType(n, 1);

    for (size_t i = 0; i < n; ++i) {
        int pivot = findPivot(A, i);
        A.element_row_transform_swap(i, pivot);
        x.element_row_transform_swap(i, pivot);

        for (size_t j = i + 1; j < n; ++j) {
            double factor = A.get_value(j, i) / A.get_value(i, i);
            for (size_t k = i; k < n; ++k) {
                A.add_value(j, k, -factor * A.get_value(i, k));
            }
            x.add_value(j, 0, -factor * x.get_value(i, 0));
        }
    }

    backsolve(A, x, x);
}

template <typename MatrixType>
void matrixSolvers<MatrixType>::LUdecomposition(MatrixType& A, MatrixType& x, const MatrixType& b) {
    size_t n = A.get_row();
    MatrixType L(n, n), U(n, n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += L.get_value(i, j) * U.get_value(j, k);
            }
            U.set_value(i, k, A.get_value(i, k) - sum);
        }

        for (size_t k = i; k < n; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += L.get_value(k, j) * U.get_value(j, i);
            }
            L.set_value(k, i, (A.get_value(k, i) - sum) / U.get_value(i, i));
        }
    }

    MatrixType y(n, 1);
    forwardSolve(L, y, b);
    backsolve(U, y, x);
}

} // namespace pnmatrix

#endif /* MATRIXSOLVERS_H_ */

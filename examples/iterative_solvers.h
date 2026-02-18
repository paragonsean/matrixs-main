#ifndef ITERATIVESOLVERS_H_
#define ITERATIVESOLVERS_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "../include/matrix.h"
#include "../include/sparse_matrix_storage.h"
#include "../include/gmres_solver.h"

#include "../include/jacobian_solver.h"
using namespace pnmatrix;
typedef matrix<sparse_matrix_storage<double>> Matrix; // Replace `std::vector<double>` with your actual container type if different

class IterativeSolvers
{
public:
    // Iterative solvers with optional error limit, max iterations, and verbose mode

    // Jacobi Method
    static int jacobi(const Matrix& A, Matrix& x_old, const Matrix& b,
                      double errLimit = 0.00001, int maxIterations = 1000, bool verbose = true);

    // Gauss-Seidel Method
    static int gaussSeidel(const Matrix& A, Matrix& x_old, const Matrix& b,
                           double errLimit = 0.00001, int maxIterations = 1000000, bool verbose = true);

    // Successive Over-Relaxation (SOR) Method
    static int sor(const double omega, const Matrix& A, Matrix& x, const Matrix& b,
                   double errLimit = 0.00001, int maxIterations = 1000000, bool verbose = false);
    double static relError(const Matrix& A, const Matrix& x, const Matrix& b);
};

#endif /* ITERATIVESOLVERS_H_ */

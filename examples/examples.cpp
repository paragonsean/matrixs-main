#include "../include/matrix.h"
#include "../include/sparse_matrix_storage.h"
#include "../include/dense_matrix_storage.h"
#include "../include/gmres_solver.h"
#include "../include/gauss_seidel_solver.h"
#include "../include/jacobian_solver.h"
#include "../include/gaussian_elimination.h" // Ensure this file exists at the specified path
#include <iostream>

using namespace pnmatrix;

// Simple print function to display matrix values with zero-based indexing
template< typename MatrixType>
void print_matrix(const MatrixType& m) {
    for(auto row = m.begin(); row != m.end(); ++row) {
        for(auto col = row.begin(); col != row.end(); ++col) {
            std::cout << "(" << col.row_index() << ", " << col.column_index() << ") " << *col << " ";
        }
        std::cout << "\n";
    }
}

// Example for GMRES method
void gmres_example() {
    matrix<sparse_matrix_storage<double>> m(3, 3);
    m.set_value(0, 0, 1);
    m.set_value(0, 1, 1);
    m.set_value(0, 2, 1);
    m.set_value(1, 0, 0);
    m.set_value(1, 1, 4);
    m.set_value(1, 2, -1);
    m.set_value(2, 0, 2);
    m.set_value(2, 1, -2);
    m.set_value(2, 2, 1);
    
    std::cout << "Example GMRES.\n";
    std::cout << "Matrix A : \n";
    print_matrix(m);
    
    matrix<sparse_matrix_storage<double>> b(3, 1);
    b.set_value(0, 0, 6);
    b.set_value(1, 0, 5);
    b.set_value(2, 0, 1);

    std::cout << "Matrix b : \n";
    print_matrix(b);
    std::cout << "Use GMRES method to solve A * x = b : \n";
    
    gmres::option op;
    op.m = 3;      
    op.rm = 1e-5;  
    std::cout << "Restart m : " << op.m << ", Error tolerance rm : " << op.rm << std::endl;
    
    gmres solver(op);
    auto result = solver.solve(m, b);
    
    std::cout << "Result : x \n";
    print_matrix(result);
    std::cout << "###\n";
}

// Example for Jacobian method
void jacobian_example() {
    matrix<sparse_matrix_storage<double>> m(3, 3);
    m.set_value(0, 0, 8);
    m.set_value(0, 1, -3);
    m.set_value(0, 2, 2);
    m.set_value(1, 0, 4);
    m.set_value(1, 1, 11);
    m.set_value(1, 2, -1);
    m.set_value(2, 0, 6);
    m.set_value(2, 1, 3);
    m.set_value(2, 2, 12);
    
    std::cout << "Example Jacobian.\n";
    std::cout << "Matrix A : \n";
    print_matrix(m);
    
    matrix<sparse_matrix_storage<double>> b(3, 1);
    b.set_value(0, 0, 20);
    b.set_value(1, 0, 33);
    b.set_value(2, 0, 36);
    
    std::cout << "Matrix b : \n";
    print_matrix(b);
    std::cout << "Use Jacobian method to solve A * x = b : \n";
    
    jacobian::option op;
    op.rm = 1e-6;
    std::cout << "rm : " << op.rm << std::endl;
    
    jacobian solver(op);
    auto result = solver.solve(m, b);
    
    std::cout << "Result : x \n";
    print_matrix(result);
    std::cout << "###\n";
}

// Example for Gauss-Seidel method
void gauss_seidel_example() {
    matrix<sparse_matrix_storage<double>> m(3, 3);
    m.set_value(0, 0, 8);
    m.set_value(0, 1, -3);
    m.set_value(0, 2, 2);
    m.set_value(1, 0, 4);
    m.set_value(1, 1, 11);
    m.set_value(1, 2, -1);
    m.set_value(2, 0, 6);
    m.set_value(2, 1, 3);
    m.set_value(2, 2, 12);
    
    std::cout << "Example Gauss-Seidel.\n";
    std::cout << "Matrix A : \n";
    print_matrix(m);
    
    matrix<sparse_matrix_storage<double>> b(3, 1);
    b.set_value(0, 0, 20);
    b.set_value(1, 0, 33);
    b.set_value(2, 0, 36);
    
    std::cout << "Matrix b : \n";
    print_matrix(b);
    std::cout << "Use Gauss-Seidel method to solve A * x = b : \n";
    
    gauss_seidel::option op;
    op.rm = 1e-6;
    std::cout << "rm : " << op.rm << std::endl;
    
    gauss_seidel solver(op);
    auto result = solver.solve(m, b);
    
    std::cout << "Result : x \n";
    print_matrix(result);
    std::cout << "###\n";
}

// Example for Gaussian Elimination with only Dense Matrix
void gaussian_elimination_example() {
    std::cout << "Example Gaussian Elimination (Dense).\n";
    
    matrix<sparse_matrix_storage<double>> m_dense(3, 3);
    m_dense.set_value(0, 0, 8);
    m_dense.set_value(0, 1, -3);
    m_dense.set_value(0, 2, 2);
    m_dense.set_value(1, 0, 4);
    m_dense.set_value(1, 1, 11);
    m_dense.set_value(1, 2, -1);
    m_dense.set_value(2, 0, 6);
    m_dense.set_value(2, 1, 3);
    m_dense.set_value(2, 2, 12);
    
    matrix<sparse_matrix_storage<double>> b_dense(3, 1);
    b_dense.set_value(0, 0, 20);
    b_dense.set_value(1, 0, 33);
    b_dense.set_value(2, 0, 36);
    
    std::cout << "Dense Matrix A : \n";
    print_matrix(m_dense);
    std::cout << "Dense Matrix b : \n";
    print_matrix(b_dense);
    
    gaussian_elimination::option op;
    op.rm = 1e-6;  // This is not used for Gaussian Elimination but kept for consistency
    std::cout << "rm : " << op.rm << std::endl;
    
    gaussian_elimination solver_dense(op);
    auto result_dense = solver_dense.solve(m_dense, b_dense);
    
    std::cout << "Dense Result : x \n";
    print_matrix(result_dense);
    std::cout << "###\n";
}

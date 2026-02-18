#include "MatrixFileHandler.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include "../include/matrix.h"
#include "../include/matrix_storage_cep.h"

using namespace pnmatrix;
typedef matrix<sparse_matrix_storage<double>> MatrixType;

void testGenerateRandomMatrixAndWriteToFile()
{
    MatrixFileHandler generator;
    std::string filename = "test_matrix.txt";

    // Redirect std::cin to simulate user input
    std::istringstream input("3\n");
    std::cin.rdbuf(input.rdbuf());

    generator.generateRandomMatrixAndWriteToFile(filename);

    // Check if file exists
    std::ifstream file(filename);
    assert(file.good());
    file.close();

    // Clean up
    std::remove(filename.c_str());
}

void testReadMatrixFromFile()
{
    MatrixFileHandler generator;
    std::string filename = "test_matrix.txt";

    // Create a test matrix and write to file
    std::ofstream file(filename);
    file << "3 3\n";
    file << "1 2 3\n4 5 6\n7 8 9\n";
    file << "14\n32\n50\n";
    file.close();

    MatrixType A(3, 3);
    MatrixType b(3, 1);

    generator.readMatrixFromFile(A, b, filename);

    // Check matrix A values
    assert(A.get_value(1, 1) == 1);
    assert(A.get_value(1, 2) == 2);
    assert(A.get_value(1, 3) == 3);
    assert(A.get_value(2, 1) == 4);
    assert(A.get_value(2, 2) == 5);
    assert(A.get_value(2, 3) == 6);
    assert(A.get_value(3, 1) == 7);
    assert(A.get_value(3, 2) == 8);
    assert(A.get_value(3, 3) == 9);

    // Check vector b values
    assert(b.get_value(1, 1) == 14);
    assert(b.get_value(2, 1) == 32);
    assert(b.get_value(3, 1) == 50);

    // Clean up
    std::remove(filename.c_str());
}

void testWriteOutSolution()
{
    MatrixFileHandler generator;
    std::string filename = "test_solution.txt";

    // Create a test solution vector
    MatrixType x(3, 1);
    x.set_value(1, 1, 1.0);
    x.set_value(2, 1, 2.0);
    x.set_value(3, 1, 3.0);

    generator.writeOutSolution(x, filename);

    // Check if file exists and content is correct
    std::ifstream file(filename);
    assert(file.good());

    double value;
    file >> value;
    assert(value == 1.0);
    file >> value;
    assert(value == 2.0);
    file >> value;
    assert(value == 3.0);

    file.close();

    // Clean up
    std::remove(filename.c_str());
}

void testWriteMatrixToFile()
{
    MatrixFileHandler generator;
    std::string filename = "test_matrix_only.txt";

    // Create a test matrix
    MatrixType A(3, 3);
    A.set_value(1, 1, 1.0);
    A.set_value(1, 2, 2.0);
    A.set_value(1, 3, 3.0);
    A.set_value(2, 1, 4.0);
    A.set_value(2, 2, 5.0);
    A.set_value(2, 3, 6.0);
    A.set_value(3, 1, 7.0);
    A.set_value(3, 2, 8.0);
    A.set_value(3, 3, 9.0);

    generator.writeMatrixToFile(A, filename);

    // Check if file exists and content is correct
    std::ifstream file(filename);
    assert(file.good());

    double value;
    file >> value;
    assert(value == 1.0);
    file >> value;
    assert(value == 2.0);
    file >> value;
    assert(value == 3.0);
    file >> value;
    assert(value == 4.0);
    file >> value;
    assert(value == 5.0);
    file >> value;
    assert(value == 6.0);
    file >> value;
    assert(value == 7.0);
    file >> value;
    assert(value == 8.0);
    file >> value;
    assert(value == 9.0);

    file.close();

    // Clean up
    std::remove(filename.c_str());
}

void testPrintMatrix()
{
    MatrixFileHandler generator;

    // Create a test matrix
    MatrixType A(3, 3);
    A.set_value(1, 1, 1.0);
    A.set_value(1, 2, 2.0);
    A.set_value(1, 3, 3.0);
    A.set_value(2, 1, 4.0);
    A.set_value(2, 2, 5.0);
    A.set_value(2, 3, 6.0);
    A.set_value(3, 1, 7.0);
    A.set_value(3, 2, 8.0);
    A.set_value(3, 3, 9.0);

    // Redirect std::cout to a stringstream to capture the output
    std::ostringstream output;
    std::streambuf* oldCout = std::cout.rdbuf(output.rdbuf());

    generator.printMatrix(A, "Test Matrix");

    // Restore std::cout
    std::cout.rdbuf(oldCout);

    // Check the output
    std::string expectedOutput = "Test Matrix (3x3):\n1 2 3 \n4 5 6 \n7 8 9 \n\n";
    assert(output.str() == expectedOutput);
}

int main()
{
    testGenerateRandomMatrixAndWriteToFile();
    testReadMatrixFromFile();
    testWriteOutSolution();
    testWriteMatrixToFile();
    testPrintMatrix();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
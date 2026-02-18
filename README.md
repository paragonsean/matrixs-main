# Sean's SPN Matrix Library

A C++17 library for solving linear equations with support for various iterative and direct methods.

## Features

- **GMRES-M** - Generalized Minimal Residual method
- **Jacobi Iteration** - Classical iterative solver
- **Gauss-Seidel Iteration** - Improved iterative method
- **Gaussian Elimination** - Direct solver
- **QR Decomposition** - Matrix factorization
- **SOR (Successive Over-Relaxation)** - Accelerated iterative method

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Running Examples
```bash
cd examples
make
./main
```

### Running Tests
```bash
cd build
make test
```

## Project Structure

- `include/` - Header files for matrix operations and solvers
- `examples/` - Example programs demonstrating library usage
- `test/` - Unit tests for all components
- `build/` - Build output directory

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher
- (Optional) Google Test for testing

## License

See LICENSE file for details.

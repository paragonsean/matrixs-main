# Sean's SPN Matrix Library

A C++17 library for solving linear equations with support for various iterative and direct methods.

## Features

- **GMRES-M** - Generalized Minimal Residual method
- **Jacobi Iteration** - Classical iterative solver
- **Gauss-Seidel Iteration** - Improved iterative method
- **Gaussian Elimination** - Direct solver
- **QR Decomposition** - Matrix factorization
- **SOR (Successive Over-Relaxation)** - Accelerated iterative method
- **Performance Benchmarking** - Comprehensive benchmarking framework
- **Parallel Processing** - Multi-threaded matrix operations
- **Memory Mapping** - High-performance file I/O for large matrices

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

### Running Performance Reports
```bash
cd build
./performance_report_example
```

This will generate comprehensive performance reports including:
- HTML report with interactive charts (`matrix_performance_report.html`)
- Text report with statistical analysis (`matrix_performance_report.txt`)
- CSV data export (`matrix_performance_report.csv`)

## Project Structure

- `include/` - Header files for matrix operations and solvers
- `examples/` - Example programs demonstrating library usage
- `test/` - Unit tests for all components
- `build/` - Build output directory

## Matrix Solver Benchmarking System

The library includes a comprehensive benchmarking system that evaluates both iterative and direct solver performance with parallel processing capabilities.

### Benchmark Architecture

![Matrix Solver Benchmarking System](https://via.placeholder.com/800x600/333333/FFFFFF?text=Matrix+Solver+Benchmarking+System+Architecture)

### Key Components

1. **Main Benchmark Driver** (`examples/main.cpp`)
   - User-driven solver selection (Gauss-Seidel vs Gaussian Elimination)
   - Automated testing across multiple matrix sizes
   - Performance timing and results aggregation

2. **Matrix Generation Pipeline** (`examples/matrix_solver.h`)
   - Parallel matrix generation using thread pools
   - Diagonal dominance for guaranteed solver convergence
   - High-performance binary file I/O with memory mapping

3. **Iterative Solver Execution** (`include/gauss_seidel_solver.h`)
   - Gauss-Seidel method with convergence checking
   - Configurable tolerance and iteration limits
   - Real-time error monitoring

4. **Direct Solver Execution** (`include/gaussian_elimination.h`)
   - Gaussian elimination with partial pivoting
   - Forward elimination and back substitution
   - Numerical stability optimization

5. **Thread Pool Operations** (`examples/thread_pool.h`)
   - Work-stealing thread pool for parallel operations
   - Custom memory allocator for task objects
   - Efficient task queuing and synchronization

### Benchmark Execution Flow

```
Benchmark Execution Flow
├── User input: solver choice
├── Main size iteration loop
│   ├── Start timing measurement
│   ├── Conditional solver selection
│   │   ├── Gauss-Seidel path → testMatrixGenerationAndSolve()
│   │   └── Gaussian Elimination path → testGaussianElimination()
│   └── End timing measurement
└── Results aggregation & logging
```

### Performance Features

- **Parallel Matrix Generation**: Multi-threaded creation of diagonally dominant matrices
- **Memory-Mapped I/O**: Zero-copy file operations for large matrices
- **High-Resolution Timing**: Nanosecond precision performance measurement
- **Statistical Analysis**: Mean, variance, and confidence intervals
- **Scalable Testing**: Support for matrix sizes from 100×100 to 5000×5000+
- **Interactive Reports**: HTML reports with charts and visualizations
- **Data Export**: CSV format for external analysis tools
- **Solver Comparison**: Automated ranking and performance analysis

### Usage Examples

```cpp
// Run comprehensive benchmark
./main

// Run performance reporting with analysis
./performance_report_example

// Test with different matrix sizes
for (size_t size : {100, 500, 1000, 2000, 5000}) {
    testMatrixGenerationAndSolve<DenseMatrix>("test.txt", "Dense", false, size);
}
```

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher
- (Optional) Google Test for testing

## License

See LICENSE file for details.

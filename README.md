# Sean's SPN Matrix Library

A high-performance C++17 library for solving linear equations with support for various iterative and direct methods, featuring sparse matrix optimization and comprehensive performance analysis.

## Features

### Solvers
- **GMRES-M** - Generalized Minimal Residual method
- **Jacobi Iteration** - Classical iterative solver
- **Gauss-Seidel Iteration** - Improved iterative method
- **Gaussian Elimination** - Direct solver
- **QR Decomposition** - Matrix factorization
- **SOR (Successive Over-Relaxation)** - Accelerated iterative method

### Matrix Storage
- **Dense Matrix Storage** - Optimized for dense matrices with contiguous memory
- **Sparse Matrix Storage** - Efficient row-based storage for sparse matrices
  - Automatic zero-element management
  - Optimized insertion and lookup operations
  - Memory-efficient for large sparse systems

### Performance & Analysis
- **Performance Benchmarking** - Comprehensive benchmarking framework
- **Statistical Analysis** - Mean, variance, confidence intervals
- **Interactive Reports** - HTML reports with Chart.js visualizations
- **Solver Comparison** - Automated ranking and performance analysis
- **Scalability Analysis** - Performance scaling across matrix sizes

### High-Performance Features
- **Parallel Processing** - Multi-threaded matrix operations
- **Memory Mapping** - High-performance file I/O for large matrices
- **Thread Pool** - Work-stealing thread pool with custom allocator
- **High-Resolution Timing** - Nanosecond precision measurements

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
./performance_report_fixed
```

This will generate comprehensive performance reports including:
- HTML report with interactive charts (`matrix_performance_report_main.html`)
- Text report with statistical analysis (`matrix_performance_report_main.txt`)
- CSV data export (`matrix_performance_report_main.csv`)

### Available Performance Reporters
- `performance_report_fixed` - Main.cpp style benchmarking (both solvers, matrix sizes 500-9000)
- `performance_report_fast` - Fast version for quick testing (smaller matrices)
- `performance_report_simple` - Simplified version without thread pool
- `performance_report_example` - Full demonstration with all features

## Project Structure

- `include/` - Header files for matrix operations and solvers
  - `matrix.h` - Core matrix interface and operations
  - `dense_matrix_storage.h` - Dense matrix storage implementation
  - `sparse_matrix_storage.h` - Sparse matrix storage with row-based optimization
  - `performance_reporter.h` - Comprehensive performance analysis framework
  - `*_solver.h` - Various solver implementations (GMRES, Gauss-Seidel, etc.)
- `examples/` - Example programs demonstrating library usage
  - `main.cpp` - Interactive benchmarking program
  - `performance_report_*.cpp` - Various performance reporting implementations
  - `matrix_*.h` - Utility classes and helpers
- `test/` - Unit tests for all components
- `build/` - Build output directory

## Matrix Storage Implementation

### Dense Matrix Storage
- **Contiguous Memory**: Optimized for cache-friendly access patterns
- **Template-Based**: Support for various numeric types
- **Direct Indexing**: O(1) element access and modification

### Sparse Matrix Storage
- **Row-Based Storage**: Each row maintains a sorted list of non-zero elements
- **Zero Management**: Configurable zero-element handling for memory optimization
- **Efficient Operations**: Optimized insertion, lookup, and traversal
- **Memory Efficient**: Ideal for large sparse systems with high sparsity ratios

```cpp
// Example: Sparse matrix with automatic zero management
SparseMatrix<double> sparse(1000, 1000);
sparse.set_value(0, 0, 5.0);  // Stores non-zero element
sparse.set_value(0, 1, 0.0);  // Can optionally store or skip zeros
```

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
// Basic matrix operations
#include "matrix.h"
#include "dense_matrix_storage.h"
#include "sparse_matrix_storage.h"

using DenseMatrix = pnmatrix::matrix<pnmatrix::dense_matrix_storage<double>>;
using SparseMatrix = pnmatrix::matrix<pnmatrix::sparse_matrix_storage<double>>;

// Create and use matrices
DenseMatrix dense(100, 100);
SparseMatrix sparse(100, 100);

// Set values
dense.set_value(0, 0, 1.0);
sparse.set_value(0, 0, 1.0);

// Solve systems
#include "gauss_seidel_solver.h"
#include "gaussian_elimination.h"

pnmatrix::gauss_seidel::option gs_opt;
gs_opt.rm = 1e-6;
pnmatrix::gauss_seidel gs_solver(gs_opt);
auto result = gs_solver.solve(dense, rhs);

// Performance reporting
#include "performance_reporter.h"
pnmatrix::benchmark::PerformanceReporter reporter;
// ... run benchmarks ...
reporter.generate_html_report("report.html", "My Analysis");
```

### Performance Analysis Features

The performance reporting system provides:

- **Interactive HTML Reports**: Chart.js-powered visualizations
- **Statistical Summaries**: Mean, median, standard deviation, confidence intervals
- **Solver Rankings**: Automated performance comparison
- **Scalability Analysis**: Performance trends across matrix sizes
- **CSV Export**: Data for external analysis tools
- **Convergence Tracking**: Iteration counts and error metrics

## Requirements

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.10 or higher
- **Google Test** (optional, for running unit tests)
- **Modern web browser** (for viewing interactive HTML reports)

## Performance Characteristics

### Matrix Size Support
- **Dense Matrices**: Up to 40,000×40,000 (limited by available memory)
- **Sparse Matrices**: Limited only by non-zero element count
- **Recommended**: 1,000×1,000 to 10,000×10,000 for optimal performance

### Solver Performance
- **Gaussian Elimination**: O(n³) complexity, best for small-to-medium dense matrices
- **Gauss-Seidel**: O(n²) per iteration, excellent for sparse, diagonally dominant systems
- **Jacobi**: Similar to Gauss-Seidel but with slower convergence
- **GMRES**: O(kn²) where k is restart size, robust for general sparse systems

### Memory Usage
- **Dense Storage**: n² × sizeof(T) bytes
- **Sparse Storage**: nnz × sizeof(node) bytes where nnz is non-zero count
- **Thread Pool**: Minimal overhead with custom allocator
- **Performance Reports**: < 1MB for typical benchmark runs

## Recent Updates

### Version 1.1 - Performance Reporting System
- ✅ Added comprehensive performance analysis framework
- ✅ Interactive HTML reports with Chart.js visualizations
- ✅ Statistical analysis and solver comparison
- ✅ Multiple performance reporting implementations
- ✅ Enhanced documentation with Doxygen-style comments
- ✅ Improved sparse matrix storage optimization
- ✅ Main.cpp style benchmarking with both solvers

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows C++17 standards
- Include appropriate unit tests
- Update documentation for new features
- Follow existing code style and naming conventions

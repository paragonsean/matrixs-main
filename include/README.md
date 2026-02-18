# Include Directory Documentation

This directory contains the core header files for the Sean's SPN Matrix Library. The library provides a template-based C++17 implementation for solving linear equations with support for both dense and sparse matrices.

## Core Components

### Matrix Classes

#### `matrix.h` 
**Main matrix class template**
- Primary interface for all matrix operations
- Template parameter: Container type (dense or sparse storage)
- Provides iterators, arithmetic operations, and solver integration
- Supports both dense and sparse matrix storage backends

### Storage Implementations

#### `dense_matrix_storage.h`
**Dense matrix storage backend**
- Contiguous memory storage for all matrix elements
- Optimized for dense matrices where most elements are non-zero
- Custom allocator support with logging capabilities
- Row-major storage layout

#### `sparse_matrix_storage.h`
**Sparse matrix storage backend**
- Compressed storage for matrices with many zero elements
- Linked list-based sparse row format
- Memory efficient for large, sparse systems
- Optimized iterator access for non-zero elements only

### Solvers

#### Iterative Solvers
- **`gmres_solver.h`** - Generalized Minimal Residual method with restart capability
- **`gauss_seidel_solver.h`** - Classical Gauss-Seidel iterative method
- **`jacobian_solver.h`** - Jacobi iteration method
- **`sor.h`** - Successive Over-Relaxation method

#### Direct Solvers
- **`gaussian_elimination.h`** - Direct Gaussian elimination with partial pivoting
- **`qr_decomposition.h`** - QR decomposition for least squares problems

### Utilities

#### `type.h`
**Basic type definitions**
- `size_type` - Standard integer type for matrix dimensions
- Foundation type used throughout the library

#### `matrix_type_traits.h`
**Type traits and metaprogramming**
- Compile-time type detection for dense vs sparse matrices
- Template metaprogramming utilities for matrix operations
- Container type tags and type checking

#### `operator_proxy.h`
**Expression templates and lazy evaluation**
- Proxy classes for matrix arithmetic operations
- Lazy evaluation to avoid temporary matrix creation
- Support for complex matrix expressions

#### `value_compare.h`
**Value comparison utilities**
- Floating-point comparison with tolerance
- Template-based value equality testing
- Used in convergence checking for iterative solvers

#### `matrix_storage_cep_config.h`
**Configuration constants**
- Compile-time configuration parameters
- Storage backend settings and thresholds

## Usage Pattern

```cpp
#include "matrix.h"
#include "dense_matrix_storage.h"
#include "sparse_matrix_storage.h"
#include "gmres_solver.h"

// Dense matrix
pnmatrix::matrix<pnmatrix::dense_matrix_storage<double>> dense_matrix(rows, cols);

// Sparse matrix  
pnmatrix::matrix<pnmatrix::sparse_matrix_storage<double>> sparse_matrix(rows, cols);

// Solver usage
pnmatrix::gmres::option opts;
opts.rm = 1e-6;
opts.m = 30;
pnmatrix::gmres solver(opts);
auto solution = solver.solve(matrix, rhs);
```

## Design Features

- **Template-based**: Compile-time optimization for different matrix types
- **Storage abstraction**: Unified interface for dense and sparse matrices
- **Expression templates**: Efficient arithmetic without temporary allocations
- **Iterator support**: STL-compatible iterators for matrix traversal
- **C++17 compliant**: Modern C++ features for better performance and safety

## Dependencies

All headers are self-contained with only standard library dependencies:
- `<cstdint>` - Fixed-width integer types
- `<vector>` - Dynamic array container
- `<algorithm>` - Standard algorithms
- `<type_traits>` - Type metaprogramming utilities
- `<memory>` - Smart pointers and allocators
- `<cmath>` - Mathematical functions
- `<cassert>` - Debug assertions

## Architecture

The library follows a layered architecture:
1. **Storage Layer** - Memory management and data layout
2. **Matrix Layer** - High-level matrix interface and operations
3. **Solver Layer** - Numerical algorithms for linear systems
4. **Utility Layer** - Type traits, comparisons, and expression templates

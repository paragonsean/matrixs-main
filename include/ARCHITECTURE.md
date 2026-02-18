# Include Directory Architecture Analysis

## Overview of File Interdependencies

The include directory implements a sophisticated template-based matrix library with a layered architecture. Each file plays a specific role in the ecosystem, creating a cohesive system for linear algebra operations.

## Core Foundation Layer

### `type.h` - Foundation
- **Purpose**: Defines basic types used throughout the library
- **Key Component**: `size_type` (int64_t) for matrix dimensions
- **Dependencies**: None (base layer)
- **Used by**: All other files

### `value_compare.h` - Numerical Utilities
- **Purpose**: Floating-point comparison with tolerance
- **Key Function**: `value_equal<T>()` - compares values with 1e-6 tolerance
- **Dependencies**: None
- **Used by**: Solvers, matrix operations for convergence checking

## Type System Layer

### `matrix_type_traits.h` - Metaprogramming Foundation
- **Purpose**: Compile-time type detection and traits
- **Key Components**:
  - `dense_container`/`sparse_container` tags
  - `is_dense_matrix`/`is_sparse_matrix` type traits
  - `is_op_type` for expression template detection
- **Dependencies**: `type.h` (indirectly)
- **Used by**: `matrix.h`, `operator_proxy.h` for type resolution

## Expression Template System

### `operator_proxy.h` - Lazy Evaluation Engine
- **Purpose**: Implements expression templates for efficient matrix operations
- **Key Classes**:
  - `op_base` - Base for all operation proxies
  - `op_add`/`op_sub`/`op_mul` - Binary operations
  - `op_mul_value`/`op_div_value` - Scalar operations
  - `op_tr` - Transpose operation
- **Template Tricks**:
  - Lazy evaluation (compute on demand)
  - Type safety via `static_assert`
  - Zero temporary allocations
- **Dependencies**: `type.h`
- **Used by**: `matrix.h` for arithmetic operations

## Storage Abstraction Layer

### `dense_matrix_storage.h` - Dense Storage Backend
- **Purpose**: Contiguous memory storage for dense matrices
- **Key Features**:
  - Row-major layout
  - Custom allocator support
  - STL-compatible iterators
- **Dependencies**: `matrix_type_traits.h`, `value_compare.h`, `type.h`
- **Used by**: End users via `matrix<dense_matrix_storage<T>>`

### `sparse_matrix_storage.h` - Sparse Storage Backend
- **Purpose**: Memory-efficient storage for sparse matrices
- **Key Features**:
  - Linked list sparse row format
  - Iterator access to non-zero elements only
  - Configurable through `matrix_storage_cep_config.h`
- **Dependencies**: `type.h`, `value_compare.h`, `matrix_type_traits.h`, `matrix_storage_cep_config.h`
- **Used by**: End users via `matrix<sparse_matrix_storage<T>>`

### `matrix_storage_cep_config.h` - Configuration
- **Purpose**: Compile-time configuration for sparse storage
- **Dependencies**: None
- **Used by**: `sparse_matrix_storage.h`

## Main Interface Layer

### `matrix.h` - Primary User Interface
- **Purpose**: Unified matrix interface for all storage types
- **Key Features**:
  - Template-based design: `matrix<Container>`
  - Expression template integration
  - Iterator support (row/column)
  - Arithmetic operations via operator proxies
  - Submatrix extraction
- **Dependencies**: All foundation files
- **Used by**: End users and all solvers

## Solver Layer

### Direct Solvers
#### `gaussian_elimination.h`
- **Purpose**: Direct solution via Gaussian elimination
- **Dependencies**: `type.h`, `value_compare.h`
- **Algorithm**: Partial pivoting elimination

#### `qr_decomposition.h`
- **Purpose**: QR factorization for least squares
- **Dependencies**: `type.h`, `value_compare.h`
- **Used by**: `gmres_solver.h` for orthogonalization

### Iterative Solvers
#### `jacobian_solver.h`
- **Purpose**: Jacobi iteration method
- **Dependencies**: `type.h`, `value_compare.h`
- **Features**: Convergence checking, singular matrix handling

#### `gauss_seidel_solver.h`
- **Purpose**: Gauss-Seidel iteration
- **Dependencies**: `type.h`, `value_compare.h`
- **Algorithm**: Improved convergence over Jacobi

#### `gmres_solver.h`
- **Purpose**: Generalized Minimal Residual method
- **Dependencies**: `type.h`, `value_compare.h`, `qr_decomposition.h`
- **Features**: Restart capability, Arnoldi process

#### `sor.h`
- **Purpose**: Successive Over-Relaxation
- **Dependencies**: `type.h`, `value_compare.h`
- **Algorithm**: Accelerated Gauss-Seidel with relaxation parameter

## Data Flow Architecture

```
User Code
    ↓
matrix.h (Interface Layer)
    ↓
┌─────────────────────┐    ┌─────────────────────────┐
│   Storage Layer     │    │   Expression Templates  │
│                     │    │                         │
│ dense_matrix_storage│    │   operator_proxy.h      │
│ sparse_matrix_storage│    │   (lazy evaluation)     │
└─────────────────────┘    └─────────────────────────┘
    ↓                              ↓
┌─────────────────────────────────────────────────────────┐
│                Foundation Layer                          │
│                                                         │
│  type.h → matrix_type_traits.h → value_compare.h       │
└─────────────────────────────────────────────────────────┘
```

## Template Metaprogramming Integration

### Type Resolution Chain
1. **Container Selection**: `matrix<dense_matrix_storage<double>>` vs `matrix<sparse_matrix_storage<double>>`
2. **Type Traits**: `is_dense_matrix`/`is_sparse_matrix` detection
3. **Expression Templates**: `op_type_flag` for lazy evaluation
4. **Solver Compatibility**: Template constraints ensure proper matrix types

### Expression Template Workflow
1. **Operation Creation**: `A + B * C` creates `op_add<op_mul, matrix>` proxy
2. **Lazy Evaluation**: No computation until `get_value()` called
3. **Type Safety**: `static_assert` ensures compatible value types
4. **Memory Efficiency**: Zero temporary matrix allocations

## Key Design Patterns

### Strategy Pattern
- Storage backends (dense vs sparse) are interchangeable strategies
- Solvers implement different algorithmic strategies

### Expression Template Pattern
- Enables natural mathematical syntax: `result = A * B + C`
- Eliminates temporary object overhead

### Template Specialization
- Compile-time optimization based on matrix type
- Type-safe operations through template constraints

### Iterator Pattern
- STL-compatible iterators for matrix traversal
- Consistent interface across storage types

## Performance Optimizations

### Compile-Time Optimizations
- Template metaprogramming eliminates runtime overhead
- Type traits enable conditional compilation
- Expression templates avoid temporary allocations

### Memory Optimizations
- Sparse storage only stores non-zero elements
- Lazy evaluation computes only requested elements
- Custom allocator support for dense matrices

### Algorithmic Optimizations
- Iterative solvers with configurable convergence criteria
- Efficient matrix multiplication in expression templates
- Zero-copy operations where possible

## Usage Patterns

### Typical User Workflow
```cpp
// 1. Choose storage type
using Matrix = pnmatrix::matrix<pnmatrix::dense_matrix_storage<double>>;

// 2. Create matrices
Matrix A(n, n), b(n, 1);

// 3. Use expression templates
Matrix C = A * A + b;  // No temporary matrices created

// 4. Solve system
pnmatrix::jacobian::option opts;
pnmatrix::jacobian solver(opts);
Matrix x = solver.solve(A, b);
```

### Solver Integration
All solvers work with any matrix type through the unified `matrix.h` interface, leveraging the storage abstraction and expression template system for optimal performance.

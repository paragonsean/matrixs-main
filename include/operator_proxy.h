#include "type.h"
#include <type_traits>
#include <cassert>

namespace pnmatrix {
/**
 * @brief Base class for all matrix operation proxies
 * 
 * This class implements the Expression Template pattern for lazy evaluation.
 * Instead of computing matrix operations immediately, it creates proxy objects
 * that represent the operation to be performed later.
 * 
 * Key benefits:
 * - Avoids temporary matrix allocations
 * - Enables operation chaining (A + B * C - D)
 * - Compile-time optimization through template metaprogramming
 */
class op_base {
private:
  size_type row_;
  size_type col_;

public:
  using op_type_flag = void;  // Marker for type traits detection
  op_base(size_type row, size_type col):row_(row), col_(col) {

  }

  op_base(const op_base&) = default;

  size_type get_row() const {
    return row_;
  }
  size_type get_column() const {
    return col_;
  }
};

/**
 * @brief Addition operation proxy for matrix addition
 * 
 * Template trick: Uses static_assert to ensure type safety at compile time.
 * This prevents adding matrices with different value types (e.g., double + float).
 * 
 * Lazy evaluation: The actual addition happens when get_value() is called,
 * not when the proxy object is created.
 */
template<typename Left, typename Right>
class op_add : public op_base {
private:
  // Compile-time type checking - ensures both matrices have same value type
  static_assert (std::is_same<typename Left::value_type, typename Right::value_type>::value,
  "invalid op_add !");
  const Left& l_;
  const Right& r_;

public:
  using value_type = typename Left::value_type;

  op_add(const Left& l, const Right& r, size_type row, size_type col):op_base(row, col), l_(l), r_(r) {

  }

  // Lazy evaluation - computes element-wise addition on demand
  value_type get_value(size_type row, size_type col) const {
    return l_.get_value(row, col) + r_.get_value(row, col);
  }
};

/**
 * @brief Matrix multiplication operation proxy
 * 
 * Template trick: Implements matrix multiplication without creating intermediate matrices.
 * The multiplication is performed element-by-element when get_value() is called.
 * 
 * Performance optimization: Only computes the specific element requested,
 * rather than computing the entire matrix product.
 */
template<typename Left, typename Right>
class op_mul : public op_base {
private:
  static_assert (std::is_same<typename Left::value_type, typename Right::value_type>::value,
  "invalid op_mul !");
  const Left& l_;
  const Right& r_;

public:
  using value_type = typename Left::value_type;

  op_mul(const Left& l, const Right& r, size_type row, size_type col):op_base(row, col), l_(l), r_(r) {
    // Runtime assertion for matrix multiplication compatibility
    assert(l_.get_column() == r_.get_row());
  }

  // Computes single element of matrix product using dot product
  // This is the core of matrix multiplication: (A*B)[i,j] = sum(A[i,k] * B[k,j])
  value_type get_value(size_type row, size_type col) const {
    value_type sum = value_type(0);
    for(size_type i = 1; i <= l_.get_column(); ++i) {
      sum += l_.get_value(row, i) * r_.get_value(i, col);
    }
    return sum;
  }
};

/**
 * @brief Subtraction operation proxy for matrix subtraction
 * 
 * Similar to op_add but implements element-wise subtraction.
 * Uses the same compile-time type checking and lazy evaluation patterns.
 */
template<typename Left, typename Right>
class op_sub : public op_base {
private:
  static_assert (std::is_same<typename Left::value_type, typename Right::value_type>::value,
  "invalid op_sub !");
  const Left& l_;
  const Right& r_;

public:
  using value_type = typename Left::value_type;

  op_sub(const Left& l, const Right& r, size_type row, size_type col):op_base(row, col), l_(l), r_(r) {

  }

  // Lazy evaluation for element-wise subtraction
  value_type get_value(size_type row, size_type col) const {
    return l_.get_value(row, col) - r_.get_value(row, col);
  }
};

/**
 * @brief Scalar division operation proxy
 * 
 * Template trick: Enables matrix / scalar operations through expression templates.
 * This allows writing A / 2.0 instead of creating a temporary matrix.
 * 
 * Performance benefit: Avoids creating intermediate matrix for scalar operations.
 */
template<typename Proxy>
class op_div_value : public op_base {
private:
  const Proxy& l_;
  typename Proxy::value_type v_;

public:
  using value_type = typename Proxy::value_type;

  op_div_value(const Proxy& l,const value_type&v, size_type row, size_type col):op_base(row, col), l_(l), v_(v) {

  }

  // Element-wise division by scalar value
  value_type get_value(size_type row, size_type col) const {
    return l_.get_value(row, col) / v_;
  }
};

/**
 * @brief Scalar multiplication operation proxy
 * 
 * Template trick: Enables matrix * scalar operations through expression templates.
 * This allows writing A * 2.0 instead of creating a temporary matrix.
 * 
 * Common use case: Scaling matrices in numerical algorithms.
 */
template<typename Proxy>
class op_mul_value : public op_base {
private:
  const Proxy& l_;
  typename Proxy::value_type v_;

public:
  using value_type = typename Proxy::value_type;

  op_mul_value(const Proxy& l,const value_type& v, size_type row, size_type col):op_base(row, col), l_(l), v_(v) {

  }

  // Element-wise multiplication by scalar value
  value_type get_value(size_type row, size_type col) const {
    return l_.get_value(row, col) * v_;
  }
};

/**
 * @brief Matrix transpose operation proxy
 * 
 * Template trick: Implements transpose without creating new matrix.
 * The transpose is computed by swapping row and column indices.
 * 
 * Memory efficiency: Zero additional memory allocation for transpose operation.
 * This is particularly useful for large sparse matrices.
 */
template<typename Proxy>
class op_tr : public op_base {
private:
  const Proxy& l_;

public:
  using value_type = typename Proxy::value_type;

  op_tr(const Proxy& l, size_type row, size_type col):op_base(row, col), l_(l) {

  }

  // Transpose operation: element at (row, col) becomes element at (col, row)
  // This is the mathematical definition of matrix transpose
  value_type get_value(size_type row, size_type col) const {
    return l_.get_value(col, row);
  }
};
}

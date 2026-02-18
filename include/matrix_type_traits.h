#pragma once
#include <type_traits>

namespace pnmatrix {
/**
 * @brief Template metaprogramming utilities for matrix type detection
 * 
 * This header contains type traits and compile-time utilities that enable
 * template metaprogramming for the matrix library. These traits allow the
 * library to:
 * - Detect matrix storage types at compile time
 * - Enable conditional compilation based on matrix properties
 * - Provide type-safe operations for different matrix types
 * - Support expression template detection and optimization
 */

/**
 * @brief Compile-time boolean AND for template parameters
 * 
 * Used for complex template constraints where multiple conditions must be true.
 * This is a utility trait for combining boolean conditions in template metaprogramming.
 * 
 * @tparam b1 First boolean condition
 * @tparam b2 Second boolean condition
 * @return std::true_type only if both conditions are true
 */
template <bool b1, bool b2>
struct both_real : std::false_type {};

/**
 * @brief Specialization for when both conditions are true
 */
template <>
struct both_real<true, true> : std::true_type {};

/**
 * @brief Tag class for dense matrix storage detection
 * 
 * This class serves as a marker type for compile-time detection of dense matrix
 * storage containers. Classes that inherit from or use this tag can be identified
 * as dense storage types through SFINAE and type traits.
 */
class dense_container {
public:
  using dense_tag = void;  ///< Marker tag for dense container detection
};

/**
 * @brief Tag class for sparse matrix storage detection
 * 
 * This class serves as a marker type for compile-time detection of sparse matrix
 * storage containers. The sparse_tag enables template specialization and conditional
 * compilation for sparse matrix operations.
 */
class sparse_container {
public:
  using sparse_tag = void;  ///< Marker tag for sparse container detection
};

/**
 * @brief Type trait for detecting expression template proxy types
 * 
 * This trait determines if a type is an operation proxy from the expression
 * template system. Operation proxies have an op_type_flag member that marks
 * them as proxy objects for lazy evaluation.
 * 
 * Uses SFINAE (Substitution Failure Is Not An Error) to detect the presence
 * of the op_type_flag member type.
 * 
 * @tparam T Type to check for operation proxy properties
 */
template <typename, typename = std::void_t<>>
struct is_op_type : std::false_type {};

/**
 * @brief Specialization for types with op_type_flag (operation proxies)
 * 
 * This specialization activates when the type T has a nested op_type_flag type,
 * indicating it's an expression template proxy object.
 * 
 * @tparam T Type that has op_type_flag member
 */
template <typename T>
struct is_op_type<T, std::void_t<typename T::op_type_flag>> : std::true_type {};

/**
 * @brief Type trait for detecting dense matrix types
 * 
 * This trait determines if a type is a matrix with dense storage. It checks
 * if the container type has a dense_tag, indicating dense matrix storage.
 * 
 * Uses the detection idiom with std::void_t to safely check for nested types.
 * 
 * @tparam T Type to check (should be matrix<Container>)
 */
template <typename, typename = std::void_t<>>
struct is_dense_matrix : std::false_type {};

/**
 * @brief Type trait for detecting sparse matrix types
 * 
 * This trait determines if a type is a matrix with sparse storage. It checks
 * if the container type has a sparse_tag, indicating sparse matrix storage.
 * 
 * @tparam T Type to check (should be matrix<Container>)
 */
template <typename, typename = std::void_t<>>
struct is_sparse_matrix : std::false_type {};

/**
 * @brief Forward declaration of matrix class
 * 
 * Required for the type trait specializations that work with matrix<T> types.
 */
template <typename T>
class matrix;

/**
 * @brief Specialization for dense matrix detection
 * 
 * This specialization activates when the container type T has a dense_tag,
 * indicating that matrix<T> uses dense storage.
 * 
 * @tparam T Container type with dense_tag
 */
template <typename T>
struct is_dense_matrix<matrix<T>, std::void_t<typename T::dense_tag>> : public std::true_type {};

/**
 * @brief Specialization for sparse matrix detection
 * 
 * This specialization activates when the container type T has a sparse_tag,
 * indicating that matrix<T> uses sparse storage.
 * 
 * @tparam T Container type with sparse_tag
 */
template <typename T>
struct is_sparse_matrix<matrix<T>, std::void_t<typename T::sparse_tag>> : public std::true_type {};

/**
 * @brief Type trait for detecting any matrix type
 * 
 * This is a general-purpose trait that determines if a type is any instantiation
 * of the matrix template. Useful for template constraints and function overloading.
 * 
 * @tparam T Type to check
 */
template <typename T>
struct is_matrix_type : std::false_type {};

/**
 * @brief Specialization for matrix types
 * 
 * This specialization matches any instantiation of matrix<T>, regardless of
 * the storage container type.
 * 
 * @tparam T Container type (dense or sparse storage)
 */
template <typename T>
struct is_matrix_type<matrix<T>> : public std::true_type {};
}

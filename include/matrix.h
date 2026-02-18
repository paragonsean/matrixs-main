#pragma once
#include "type.h"
#include "operator_proxy.h"
#include "value_compare.h"
#include "matrix_type_traits.h"
#include <functional>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <memory>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace pnmatrix {
template<class Container>
class matrix {
public:
  using container_type = Container;
  using value_type = typename container_type::value_type;
  using row_iterator = typename container_type::row_iterator;
  using column_iterator = typename container_type::row_iterator::column_iterator;
  using const_row_iterator = typename container_type::const_row_iterator;
  using const_column_iterator = typename container_type::const_row_iterator::const_column_iterator;

  matrix() = default;

  matrix(size_type row, size_type column) : container_(row, column) {}

  ~matrix() = default;

  matrix(const matrix& other) : container_(other.container_) {}

  matrix(matrix&& other) : container_(std::move(other.container_)) {}

  template <typename Proxy>
  matrix(const Proxy& pro);

  value_type operator()(size_type row, size_type column) const {
    range_check(row, column);
    return get_value(row, column);
  }

  matrix& operator=(const matrix& other) {
    container_ = other.container_;
    return *this;
  }

  matrix& operator=(matrix&& other) {
    container_ = std::move(other.container_);
    return *this;
  }

  bool operator==(const matrix& other) const {
    if (get_row() != other.get_row() || get_column() != other.get_column()) {
      return false;
    }
    for (auto row = begin(); row != end(); ++row) {
      for (auto col = row.begin(); col != row.end(); ++col) {
        if (!value_equal(*col, other.get_value(col.row_index(), col.column_index()))) {
          return false;
        }
      }
    }
    return true;
  }

  bool operator!=(const matrix& other) const {
    return !(*this == other);
  }

  // Operator[] for direct indexed access
  value_type& operator[](std::pair<size_t, size_t> index) {
    size_t row = index.first;
    size_t col = index.second;
    range_check(row, col);
    return get_value(row, col);
  }

  const value_type& operator[](std::pair<size_t, size_t> index) const {
    size_t row = index.first;
    size_t col = index.second;
    range_check(row, col);
    return get_value(row, col);
  }

  inline size_type get_row() const {
    return container_.get_row();
  }

  inline size_type get_column() const {
    return container_.get_column();
  }

  matrix get_sub_matrix(size_type row_begin, size_type r, size_type col_begin, size_type c) const {
    // Extracts a submatrix of dimension r x c starting at (row_begin, col_begin)
    // Both row_begin and col_begin are zero-based
    matrix result(r, c);
    for (auto row = begin(); row != end(); ++row) {
      for (auto col = row.begin(); col != row.end(); ++col) {
        if (col.row_index() < row_begin || col.column_index() < col_begin) {
          continue;
        }
        if (col.row_index() >= row_begin + r || col.column_index() >= col_begin + c) {
          continue;
        }
        size_type rr = col.row_index() - row_begin;
        size_type cc = col.column_index() - col_begin;
        result.set_value_withoutcheck(rr, cc, *col);
      }
    }
    return result;
  }

  void set_value(size_type row, size_type column, const value_type& value) {
    range_check(row, column);
    set_value_withoutcheck(row, column, value);
  }

  void set_value_from_matrix(size_type row_begin, size_type column_begin, const matrix& m) {
    // Insert matrix m into *this at position (row_begin, column_begin)
    // All indices are zero-based
    range_check(row_begin + m.get_row() - 1, column_begin + m.get_column() - 1);

    // Clear the corresponding region first (if needed)
    for (size_type i = row_begin; i < row_begin + m.get_row(); ++i) {
      for (size_type j = column_begin; j < column_begin + m.get_column(); ++j) {
        set_value_withoutcheck(i, j, value_type(0));
      }
    }

    // Copy values from m
    for (auto row = m.begin(); row != m.end(); ++row) {
      for (auto col = row.begin(); col != row.end(); ++col) {
        size_type r = row_begin + col.row_index();
        size_type c = column_begin + col.column_index();
        set_value_withoutcheck(r, c, *col);
      }
    }
  }

  void set_column(size_type column_begin, const matrix& mat) {
    // Sets a column (at column_begin) to values from mat (which should be a single column matrix)
    column_range_check(column_begin);
    assert(mat.get_column() == 1);
    set_value_from_matrix(0, column_begin, mat);
  }

  matrix get_nth_column(size_type n) const {
    // Extract a single column (n-th column) as a matrix
    return get_sub_matrix(0, get_row(), n, 1);
  }

  void add_value(size_type row, size_type column, const value_type& value) {
    range_check(row, column);
    add_value_withoutcheck(row, column, value);
  }

  value_type get_value(size_type row, size_type column) const {
    range_check(row, column);
    return get_value_withoutcheck(row, column);
  }

  row_iterator begin() {
    return container_.begin();
  }

  row_iterator end() {
    return container_.end();
  }

  const_row_iterator begin() const {
    return container_.begin();
  }

  const_row_iterator end() const {
    return container_.end();
  }

  bool inverse_with_ert(matrix& result) {
    // Compute matrix inverse using elementary row transformations
    // Assumes matrix is square
    assert(get_row() == get_column());
    if (get_row() == 1 && get_column() == 1) {
      result = matrix(1, 1);
      result.set_value(0, 0, 1.0 / get_value(0, 0));
      return true;
    }
    matrix tmpStorage = *this;
    result = get_identity_matrix(get_row());

    // Forward elimination
    for (size_type i = 0; i < get_row(); ++i) {
      if (value_equal(get_value_withoutcheck(i, i), value_type(0))) {
        bool error = true;
        for (size_type j = i + 1; j < get_row(); ++j) {
          if (!value_equal(get_value_withoutcheck(j, i), value_type(0))) {
            element_row_transform_swap(i, j);
            result.element_row_transform_swap(i, j);
            error = false;
            break;
          }
        }
        if (error) return false;
      }

      for (size_type j = i + 1; j < get_row(); ++j) {
        value_type j_i = get_value_withoutcheck(j, i);
        if (!value_equal(j_i, value_type(0))) {
          value_type k = -j_i / get_value(i, i);
          element_row_transform_plus(j, i, k);
          result.element_row_transform_plus(j, i, k);
        }
      }
      value_type k = value_type(1.0) / get_value(i, i);
      element_row_transform_multi(i, k);
      result.element_row_transform_multi(i, k);
    }

    // Back substitution
    for (size_type i = get_row() - 1; i > 0; --i) {
      for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
        value_type k = -get_value(j, i);
        element_row_transform_plus(j, i, k);
        result.element_row_transform_plus(j, i, k);
      }
    }

    *this = tmpStorage;
    return true;
  }

  value_type get_vector_second_norm() const {
    // Assumes a single-column vector
    assert(get_column() == 1);
    value_type sum = value_type(0);
    for (auto row_iter = begin(); row_iter != end(); ++row_iter) {
      for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
        sum += (*colu_iter) * (*colu_iter);
      }
    }
    return std::sqrt(sum);
  }

  value_type get_vector_inner_product(const matrix& m) const {
    // Assumes current matrix is 1 x N and m is N x 1
    assert(get_row() == 1 && m.get_column() == 1);
    assert(get_column() == m.get_row());
    value_type sum = value_type(0);
    const_row_iterator row_iter = begin();
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      sum += *colu_iter * m.get_value_withoutcheck(colu_iter.column_index(), 0);
    }
    return sum;
  }

  size_type get_nth_row_size(size_type row) const {
    row_range_check(row);
    return container_.get_nth_row_size(row);
  }

  void resize(size_type row, size_type column) {
    if (row == get_row() && column == get_column()) {
      return;
    }
    assert(row >= 1 && column >= 1);
    container_.resize(row, column);
  }

  void every_nozero_element(const std::function<void(const_column_iterator iterator)>& func) const {
    for (auto row_iter = begin(); row_iter != end(); ++row_iter) {
      for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
        if (value_equal(*colu_iter, value_type(0))) {
          continue;
        }
        func(colu_iter);
      }
    }
  }

  void every_nozero_element(const std::function<void(column_iterator iterator)>& func) {
    for (auto row_iter = begin(); row_iter != end(); ++row_iter) {
      for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
        if (value_equal(*colu_iter, value_type(0))) {
          continue;
        }
        func(colu_iter);
      }
    }
  }

  void element_row_transform_swap(size_type row_i, size_type row_j) {
    row_range_check(row_i);
    row_range_check(row_j);
    container_.element_row_transform_swap(row_i, row_j);
  }

  void element_row_transform_multi(size_type row, value_type k) {
    row_range_check(row);
    container_.element_row_transform_multi(row, k);
  }

  void element_row_transform_plus(size_type row_i, size_type row_j, value_type k) {
    row_range_check(row_i);
    row_range_check(row_j);
    container_.element_row_transform_plus(row_i, row_j, k);
  }

  static matrix get_identity_matrix(size_type row) {
    matrix result(row, row);
    for (size_type i = 0; i < row; ++i) {
      result.set_value_withoutcheck(i, i, value_type(1));
    }
    return result;
  }

  size_type get_element_count() const {
    return container_.get_element_count();
  }

  void delete_row(size_type row) {
    assert(row >= 0 && row < get_row());
    container_.delete_row(row);
  }

  void delete_column(size_type column) {
    assert(column >= 0 && column < get_column());
    container_.delete_column(column);
  }

  void readFromFile(const std::string& filename) {
    std::ifstream input(filename);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    size_type rows, cols;
    input >> rows >> cols; // Assuming the file specifies rows and columns
    if (rows <= 0 || cols <= 0) {
      throw std::runtime_error("Invalid matrix dimensions in file.");
    }

    resize(rows, cols);

    value_type value;
    for (size_type i = 0; i < rows; ++i) {
      for (size_type j = 0; j < cols; ++j) {
        if (!(input >> value)) {
          throw std::runtime_error("Error reading matrix data from file.");
        }
        set_value(i, j, value);
      }
    }

    input.close();
  }

private:
  Container container_;

  explicit matrix(Container&& container) : container_(std::move(container)) {}

  inline void range_check(size_type row, size_type column) const {
    assert(row >= 0 && column >= 0 && row < get_row() && column < get_column());
  }

  inline void row_range_check(size_type row) const {
    assert(row >= 0 && row < get_row());
  }

  inline void column_range_check(size_type column) const {
    assert(column >= 0 && column < get_column());
  }

  inline void set_value_withoutcheck(size_type row, size_type column, const value_type& value) {
    container_.set_value(row, column, value);
  }

  inline void add_value_withoutcheck(size_type row, size_type column, const value_type& value) {
    container_.add_value(row, column, value);
  }

  inline value_type get_value_withoutcheck(size_type row, size_type column) const {
    return container_.get_value(row, column);
  }
};

template<typename Proxy1, typename Proxy2,
  typename std::enable_if<
    both_real<
      std::disjunction<is_op_type<Proxy1>, is_dense_matrix<Proxy1>>::value,
      std::disjunction<is_op_type<Proxy2>, is_dense_matrix<Proxy2>>::value
    >::value, int>::type = 0>
auto operator+(const Proxy1& m1, const Proxy2& m2)->op_add<Proxy1, Proxy2> {
  assert(m1.get_row() == m2.get_row());
  assert(m1.get_column() == m2.get_column());
  return op_add<Proxy1, Proxy2>(m1, m2, m1.get_row(), m1.get_column());
}

template<typename MatrixType, typename std::enable_if<is_sparse_matrix<MatrixType>::value, int>::type = 0>
auto operator+(const MatrixType& m1, const MatrixType& m2)->MatrixType {
  assert(m1.get_row() == m2.get_row() && m1.get_column() == m2.get_column());
  MatrixType result(m1.get_row(), m1.get_column());
  for (auto row_iter = m1.begin(); row_iter != m1.end(); ++row_iter) {
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      result.set_value(colu_iter.row_index(), colu_iter.column_index(), *colu_iter);
    }
  }
  for (auto row_iter = m2.begin(); row_iter != m2.end(); ++row_iter) {
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      result.add_value(colu_iter.row_index(), colu_iter.column_index(), *colu_iter);
    }
  }
  return result;
}

template<typename Proxy1, typename Proxy2,
  typename std::enable_if<
    both_real<
      std::disjunction<is_op_type<Proxy1>, is_dense_matrix<Proxy1>>::value,
      std::disjunction<is_op_type<Proxy2>, is_dense_matrix<Proxy2>>::value
    >::value, int>::type = 0>
auto operator-(const Proxy1& m1, const Proxy2& m2)->op_sub<Proxy1, Proxy2> {
  assert(m1.get_row() == m2.get_row());
  assert(m1.get_column() == m2.get_column());
  return op_sub<Proxy1, Proxy2>(m1, m2, m1.get_row(), m1.get_column());
}

template<typename MatrixType, typename std::enable_if<is_sparse_matrix<MatrixType>::value, int>::type = 0>
auto operator-(const MatrixType& m1, const MatrixType& m2)->MatrixType {
  assert(m1.get_row() == m2.get_row() && m1.get_column() == m2.get_column());
  MatrixType result(m1.get_row(), m1.get_column());
  for (auto row_iter = m1.begin(); row_iter != m1.end(); ++row_iter) {
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      result.set_value(colu_iter.row_index(), colu_iter.column_index(), *colu_iter);
    }
  }
  for (auto row_iter = m2.begin(); row_iter != m2.end(); ++row_iter) {
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      result.add_value(colu_iter.row_index(), colu_iter.column_index(), -(*colu_iter));
    }
  }
  return result;
}

template <class Container>
std::ostream& operator<<(std::ostream& out, const matrix<Container>& m) {
  const size_t row_count = m.get_row();
  const size_t column_count = m.get_column();

  out << std::fixed << std::setprecision(4);

  for (size_t i = 0; i < row_count; ++i) {
    for (size_t j = 0; j < column_count; ++j) {
      out << std::setw(10) << m.get_value(i, j);
    }
    out << '\n';
  }
  return out;
}

template<typename Proxy1, typename Proxy2,
  typename std::enable_if<
    both_real<
      std::disjunction<is_op_type<Proxy1>, is_dense_matrix<Proxy1>>::value,
      std::disjunction<is_op_type<Proxy2>, is_dense_matrix<Proxy2>>::value
    >::value, int>::type = 0>
auto operator*(const Proxy1& m1, const Proxy2& m2)->op_mul<Proxy1, Proxy2> {
  assert(m1.get_column() == m2.get_row());
  return op_mul<Proxy1, Proxy2>(m1, m2, m1.get_row(), m2.get_column());
}

template<typename MatrixType, typename MatrixType2,
  typename std::enable_if<
    std::conjunction<is_sparse_matrix<MatrixType>, is_matrix_type<MatrixType2>>::value,
  int>::type = 0>
auto operator*(const MatrixType& m1, const MatrixType2& m2)->MatrixType2 {
  assert(m1.get_column() == m2.get_row());
  using value_type = typename MatrixType::value_type;
  static_assert(std::is_same<value_type, typename MatrixType2::value_type>::value, "error.");

  MatrixType2 result(m1.get_row(), m2.get_column());
  for (auto row = m1.begin(); row != m1.end(); ++row) {
    for (size_type i = 0; i < m2.get_column(); ++i) {
      value_type sum = value_type(0);
      size_type row_ = row.row_index();
      size_type colu_ = i;
      for (auto col = row.begin(); col != row.end(); ++col) {
        sum += *col * m2.get_value(col.column_index(), i);
      }
      result.set_value(row_, colu_, sum);
    }
  }
  return result;
}

template<typename MatrixType, typename std::enable_if<is_sparse_matrix<MatrixType>::value, int>::type = 0>
MatrixType operator/(const MatrixType& m, typename MatrixType::value_type value) {
  MatrixType result(m.get_row(), m.get_column());
  m.every_nozero_element([&](typename MatrixType::const_column_iterator iter)->void {
    result.set_value(iter.row_index(), iter.column_index(), *iter / value);
  });
  return result;
}

template<typename MatrixType, typename std::enable_if<is_sparse_matrix<MatrixType>::value, int>::type = 0>
MatrixType operator*(const MatrixType& m, typename MatrixType::value_type value) {
  MatrixType result(m.get_row(), m.get_column());
  m.every_nozero_element([&](typename MatrixType::const_column_iterator iter)->void {
    result.set_value(iter.row_index(), iter.column_index(), *iter * value);
  });
  return result;
}

template<typename Proxy, typename std::enable_if<
  std::disjunction<is_op_type<Proxy>, is_dense_matrix<Proxy>>::value, int>::type = 0>
auto operator/(const Proxy& m, typename Proxy::value_type value)->op_div_value<Proxy> {
  return op_div_value<Proxy>(m, value, m.get_row(), m.get_column());
}

template<typename Proxy, typename std::enable_if<
  std::disjunction<is_op_type<Proxy>, is_dense_matrix<Proxy>>::value, int>::type = 0>
auto operator*(const Proxy& m, typename Proxy::value_type value)->op_mul_value<Proxy> {
  return op_mul_value<Proxy>(m, value, m.get_row(), m.get_column());
}

template<typename Proxy, typename std::enable_if<
  std::disjunction<is_op_type<Proxy>, is_dense_matrix<Proxy>>::value, int>::type = 0>
op_tr<Proxy> tr(const Proxy& m) {
  return op_tr<Proxy>(m, m.get_column(), m.get_row());
}

template<typename MatrixType, typename std::enable_if<is_sparse_matrix<MatrixType>::value, int>::type = 0>
MatrixType tr(const MatrixType& m) {
  MatrixType result(m.get_column(), m.get_row());
  for (auto row_iter = m.begin(); row_iter != m.end(); ++row_iter) {
    for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
      result.set_value(colu_iter.column_index(), colu_iter.row_index(), *colu_iter);
    }
  }
  return result;
}

template <typename MatrixType, typename Proxy, typename std::enable_if<
  is_dense_matrix<MatrixType>::value, int>::type = 0>
void construct_from_proxy(MatrixType& self, const Proxy& pro) {
  for (size_type i = 0; i < self.get_row(); ++i) {
    for (size_type j = 0; j < self.get_column(); ++j) {
      self.set_value(i, j, pro.get_value(i, j));
    }
  }
}

template <typename Container>
template <typename Proxy>
matrix<Container>::matrix(const Proxy& pro) : container_(pro.get_row(), pro.get_column()) {
  construct_from_proxy(*this, pro);
}

} // namespace pnmatrix

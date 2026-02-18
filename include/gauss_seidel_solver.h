#pragma once

#include "type.h"
#include "value_compare.h"
#include <utility>
#include <cassert>
#include <cmath>
#include <iostream>

namespace pnmatrix {
class gauss_seidel {
private:
  double rm_;

public:
  struct option {
    double rm = 1e-6;
  };

  gauss_seidel(option op): rm_(op.rm) {}

  template<class MatrixType>
  MatrixType solve(MatrixType& coeff, MatrixType& b) {
    assert(coeff.get_column() == b.get_row() && b.get_column() == 1);
    size_type x_count = coeff.get_column();
    MatrixType x_prev(x_count, 1);
    MatrixType x_next(x_count, 1);

    size_t iteration_count = 1;
    while (true) {
      for (auto row_iter = coeff.begin(); row_iter != coeff.end(); ++row_iter) {
        double sum = 0.0;
        size_type row = row_iter.row_index();

        for (auto colu_iter = row_iter.begin(); colu_iter != row_iter.end(); ++colu_iter) {
          size_type col = colu_iter.column_index();
          double val = *colu_iter;
          if (col < row) {
            sum += val * x_next.get_value(col, 0);
          } else if (col > row) {
            sum += val * x_prev.get_value(col, 0);
          }
        }

        double rhs = b.get_value(row, 0) - sum;
        double diag = coeff.get_value(row, row);
        double result = (value_equal(diag, 0.0)) ? 0.0 : (rhs / diag);

        x_next.set_value(row, 0, result);
      }

      double max_err = max_error(x_prev, x_next);
      if (max_err <= rm_) {
        break;
      } else {
        std::swap(x_prev, x_next);
        ++iteration_count;
      }
    }
    return x_next;
  }

private:
  template<class MatrixType>
  double max_error(const MatrixType& m1, const MatrixType& m2) {
    assert(m1.get_column() == m2.get_column() && m1.get_row() == m2.get_row());

    double max_err = 0.0;
    for (size_type row = 0; row < m1.get_row(); ++row) {
      for (size_type col = 0; col < m1.get_column(); ++col) {
        double error = std::abs(m1.get_value(row, col) - m2.get_value(row, col));
        if (error > max_err) {
          max_err = error;
        }
      }
    }
    return max_err;
  }
};
}

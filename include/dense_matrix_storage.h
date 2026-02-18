// matrix_storage_block.h
#pragma once

#include "matrix_type_traits.h"
#include "value_compare.h"
#include "type.h"
#include <vector>
#include <cassert>
#include <memory>
#include <algorithm>

namespace pnmatrix {

// Forward declaration of custom allocator (if needed)
template <typename T>
class logging_allocator;


// Existing matrix_storage_block class templated on ValueType and Allocator
template <typename ValueType, typename Allocator = std::allocator<ValueType>>
class dense_matrix_storage : public dense_container {
public:
    using value_type = ValueType;
    using allocator_type = Allocator;
    using size_type = std::size_t;

    using allocator_traits = std::allocator_traits<Allocator>;
    using pointer = typename allocator_traits::pointer;
    using const_pointer = typename allocator_traits::const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;
    using difference_type = typename allocator_traits::difference_type;

    // Rebind struct (optional in C++11 and later)
    template <typename U>
    struct rebind {
        using other = dense_matrix_storage<U, typename allocator_traits::template rebind_alloc<U>>;
    };

    // Default Constructor
    dense_matrix_storage()
    : my_row_(0), my_column_(0), block_(Allocator()), block_row_(0), block_column_(0) {}

    // Constructor with dimensions and optional allocator
    dense_matrix_storage(size_type row, size_type column, const Allocator& alloc = Allocator())
        : my_row_(row), my_column_(column), block_row_(row), block_column_(column), block_(alloc)
    {
        assert(row > 0 && column > 0);
        block_.resize(row * column, ValueType(0));
    }

    // Copy Constructor
    dense_matrix_storage(const dense_matrix_storage& other)
        : my_row_(other.my_row_), my_column_(other.my_column_),
          block_row_(other.block_row_), block_column_(other.block_column_),
          block_(allocator_traits::select_on_container_copy_construction(other.block_.get_allocator()))
    {
        block_.resize(my_row_ * my_column_);
        for (size_type i = 0; i < my_row_; ++i) {
            for (size_type j = 0; j < my_column_; ++j) {
                set_value(i, j, other.get_value(i, j));
            }
        }
    }

    // Move Constructor
    dense_matrix_storage(dense_matrix_storage&& other) noexcept
        : my_row_(other.my_row_), my_column_(other.my_column_),
          block_row_(other.block_row_), block_column_(other.block_column_),
          block_(std::move(other.block_))
    {
        other.my_row_ = 0;
        other.my_column_ = 0;
        other.block_row_ = 0;
        other.block_column_ = 0;
    }

    // Copy Assignment Operator
    dense_matrix_storage& operator=(const dense_matrix_storage& other) {
        if (this != &other) {
            if (allocator_traits::propagate_on_container_copy_assignment::value && 
                block_.get_allocator() != other.block_.get_allocator()) {
                block_ = other.block_;
            } else {
                for (size_type i = 0; i < std::min(my_row_, other.my_row_); ++i) {
                    for (size_type j = 0; j < std::min(my_column_, other.my_column_); ++j) {
                        set_value(i, j, other.get_value(i, j));
                    }
                }
            }
            my_row_ = other.my_row_;
            my_column_ = other.my_column_;
            block_row_ = other.block_row_;
            block_column_ = other.block_column_;
        }
        return *this;
    }

    // Move Assignment Operator
    dense_matrix_storage& operator=(dense_matrix_storage&& other) noexcept {
        if (this != &other) {
            block_ = std::move(other.block_);
            my_row_ = other.my_row_;
            my_column_ = other.my_column_;
            block_row_ = other.block_row_;
            block_column_ = other.block_column_;
            
            other.my_row_ = 0;
            other.my_column_ = 0;
            other.block_row_ = 0;
            other.block_column_ = 0;
        }
        return *this;
    }

    // Equality operators
    bool operator==(const dense_matrix_storage& other) const {
        if (get_row() != other.get_row() || get_column() != other.get_column()) {
            return false;
        }
        for (size_type i = 0; i < my_row_; ++i) {
            for (size_type j = 0; j < my_column_; ++j) {
                if (!value_equal(get_value(i, j), other.get_value(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    bool operator!=(const dense_matrix_storage& other) const {
        return !(*this == other);
    }

    // Set and add value methods
    void set_value(size_type row, size_type column, const value_type& v) {
        block_[get_index(row, column)] = v;
    }

    void add_value(size_type row, size_type column, const value_type& v) {
        block_[get_index(row, column)] += v;
    }

    value_type get_value(size_type row, size_type column) const {
        return block_[get_index(row, column)];
    }

    // Getters for dimensions
    inline size_type get_row() const {
        return my_row_;
    }

    inline size_type get_column() const {
        return my_column_;
    }

    // Get row size (always my_column_ for dense storage)
    size_type get_nth_row_size(size_type row) const {
        (void)row; // Unused parameter
        return my_column_;
    }

    // Delete a row by shifting subsequent rows up
    void delete_row(size_type row) {
        assert(row < my_row_);
        // Move all rows after 'row' one step up
        for (size_type i = row + 1; i < my_row_; ++i) {
            for (size_type j = 0; j < my_column_; ++j) {
                set_value(i - 1, j, get_value(i, j));
            }
        }
        --my_row_;
        block_.resize(my_row_ * my_column_);
    }

    // Delete a column by shifting subsequent columns left
    void delete_column(size_type column) {
        assert(column < my_column_);
        // Move all columns after 'column' one step left
        for (size_type i = 0; i < my_row_; ++i) {
            for (size_type j = column + 1; j < my_column_; ++j) {
                set_value(i, j - 1, get_value(i, j));
            }
        }
        --my_column_;
        // Resize block to new size
        std::vector<ValueType, Allocator> new_block(my_row_ * my_column_, ValueType(0));
        for (size_type i = 0; i < my_row_; ++i) {
            for (size_type j = 0; j < my_column_; ++j) {
                new_block[i * my_column_ + j] = get_value(i, j);
            }
        }
        block_.swap(new_block);
    }

    // Resize the matrix
    void resize(size_type new_row, size_type new_column) {
        assert(new_row > 0 && new_column > 0);
        dense_matrix_storage tmp(new_row, new_column, allocator_traits::select_on_container_copy_construction(block_.get_allocator()));
        for (size_type i = 0; i < std::min(my_row_, new_row); ++i) {
            for (size_type j = 0; j < std::min(my_column_, new_column); ++j) {
                tmp.set_value(i, j, get_value(i, j));
            }
        }
        block_.swap(tmp.block_);
        my_row_ = new_row;
        my_column_ = new_column;
        block_row_ = new_row;
        block_column_ = new_column;
    }

    // Iterators for rows and columns
    class row_iterator {
    private:
        dense_matrix_storage<ValueType, Allocator>* handle_;
        size_type row_index_;

    public:
        row_iterator(dense_matrix_storage<ValueType, Allocator>* h, size_type r)
            : handle_(h), row_index_(r) {
        }

        row_iterator& operator++() {
            ++row_index_;
            return *this;
        }

        row_iterator operator++(int) {
            row_iterator result = *this;
            ++(*this);
            return result;
        }

        bool operator==(const row_iterator& other) const {
            return handle_ == other.handle_ && row_index_ == other.row_index_;
        }

        bool operator!=(const row_iterator& other) const {
            return !(*this == other);
        }

        size_type row_index() const {
            return row_index_;
        }

        class column_iterator {
        private:
            dense_matrix_storage<ValueType, Allocator>* handle_;
            size_type row_;
            size_type column_;

        public:
            column_iterator(dense_matrix_storage<ValueType, Allocator>* h, size_type r, size_type c)
                : handle_(h), row_(r), column_(c) {
            }

            column_iterator& operator++() {
                ++column_;
                return *this;
            }

            column_iterator operator++(int) {
                column_iterator result = *this;
                ++(*this);
                return result;
            }

            bool operator==(const column_iterator& other) const {
                return handle_ == other.handle_ && row_ == other.row_ && column_ == other.column_;
            }

            bool operator!=(const column_iterator& other) const {
                return !(*this == other);
            }

            value_type& operator*() {
                return handle_->block_[handle_->get_index(row_, column_)];
            }

            value_type* operator->() {
                return &(operator*());
            }

            size_type column_index() const {
                return column_;
            }

            size_type row_index() const {
                return row_;
            }
        };

        column_iterator begin() {
            return column_iterator(handle_, row_index_, 0);
        }

        column_iterator end() {
            return column_iterator(handle_, row_index_, handle_->get_column());
        }
    };

    class const_row_iterator {
    private:
        const dense_matrix_storage<ValueType, Allocator>* const handle_;
        size_type row_index_;

    public:
        const_row_iterator(const dense_matrix_storage<ValueType, Allocator>* const h, size_type r)
            : handle_(h), row_index_(r) {
        }

        const_row_iterator& operator++() {
            ++row_index_;
            return *this;
        }

        const_row_iterator operator++(int) {
            const_row_iterator result = *this;
            ++(*this);
            return result;
        }

        bool operator==(const const_row_iterator& other) const {
            return handle_ == other.handle_ && row_index_ == other.row_index_;
        }

        bool operator!=(const const_row_iterator& other) const {
            return !(*this == other);
        }

        size_type row_index() const {
            return row_index_;
        }

        class const_column_iterator {
        private:
            const dense_matrix_storage<ValueType, Allocator>* handle_;
            size_type row_;
            size_type column_;

        public:
            const_column_iterator(const dense_matrix_storage<ValueType, Allocator>* h, size_type r, size_type c)
                : handle_(h), row_(r), column_(c) {
            }

            const_column_iterator& operator++() {
                ++column_;
                return *this;
            }

            const_column_iterator operator++(int) {
                const_column_iterator result = *this;
                ++(*this);
                return result;
            }

            bool operator==(const const_column_iterator& other) const {
                return handle_ == other.handle_ && row_ == other.row_ && column_ == other.column_;
            }

            bool operator!=(const const_column_iterator& other) const {
                return !(*this == other);
            }

            const value_type& operator*() const {
                return handle_->block_[handle_->get_index(row_, column_)];
            }

            const value_type* operator->() const {
                return &(operator*());
            }

            size_type column_index() const {
                return column_;
            }

            size_type row_index() const {
                return row_;
            }
        };

        const_column_iterator begin() const {
            return const_column_iterator(handle_, row_index_, 0);
        }

        const_column_iterator end() const {
            return const_column_iterator(handle_, row_index_, handle_->get_column());
        }
    };

    // Begin and end iterators for rows
    row_iterator begin() {
        return row_iterator(this, 0);
    }

    row_iterator end() {
        return row_iterator(this, my_row_);
    }

    const_row_iterator begin() const {
        return const_row_iterator(this, 0);
    }

    const_row_iterator end() const {
        return const_row_iterator(this, my_row_);
    }

    // Row transformation methods
    void element_row_transform_swap(size_type row_i, size_type row_j) {
        assert(row_i < my_row_ && row_j < my_row_);
        for (size_type j = 0; j < my_column_; ++j) {
            std::swap(block_[get_index(row_i, j)], block_[get_index(row_j, j)]);
        }
    }

    void element_row_transform_multi(size_type row, value_type k) {
        assert(row < my_row_);
        for (size_type j = 0; j < my_column_; ++j) {
            block_[get_index(row, j)] *= k;
        }
    }

    void element_row_transform_plus(size_type row_i, size_type row_j, value_type k) {
        assert(row_i < my_row_ && row_j < my_row_);
        for (size_type j = 0; j < my_column_; ++j) {
            add_value(row_i, j, get_value(row_j, j) * k);
        }
    }

    // Allocator Accessor
    Allocator get_allocator() const noexcept {
        return block_.get_allocator();
    }

private:
    size_type my_row_;
    size_type my_column_;
    std::vector<ValueType, Allocator> block_;
    size_type block_row_;
    size_type block_column_;

    // Helper function to calculate 1D index from 2D coordinates
    inline size_type get_index(size_type row, size_type column) const {
        assert(row < block_row_ && column < block_column_);
        return row * block_column_ + column;
    }
};

} // namespace pnmatrix

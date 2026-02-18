// matrix_storage_cep.h
#pragma once
#ifndef MATRIX_STORAGE_CEP_H
#define MATRIX_STORAGE_CEP_H

#include "type.h"
#include "value_compare.h"
#include "matrix_type_traits.h"
#include "matrix_storage_cep_config.h"
#include <vector>
#include <cassert>
#include <algorithm>
#include <memory>

namespace pnmatrix {

template<class ValueType, class Allocator = std::allocator<ValueType>>
class sparse_matrix_storage : public sparse_container {
public:
    using value_type = ValueType;
    using allocator_type = Allocator;
    using self = sparse_matrix_storage<ValueType, Allocator>;

private:
    struct node_ {
        size_type column_;
        value_type value_;
        node_(size_type c, value_type v) : column_(c), value_(v) {}
    };

    struct each_row_container_ {
        using node_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<node_>;
        std::vector<node_, node_allocator_type> this_row_;
    };

    using row_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<each_row_container_>;
    std::vector<each_row_container_, row_allocator_type> container_;

    size_type my_row_;
    size_type my_column_;
    size_type element_count_;

public:
    sparse_matrix_storage(size_type row, size_type column, const Allocator& alloc = Allocator())
        : my_row_(row),
          my_column_(column),
          element_count_(0),
          container_(alloc) // Use the provided allocator
    {
        if (row <= 0 || column <= 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero");
        }

        container_.resize(row*column);
        // Potentially reserve space for a sparse structure
        container_.reserve(row * column);
    }
    sparse_matrix_storage()
    : my_row_(0), my_column_(0), element_count_(0), container_() {}

    ~sparse_matrix_storage() = default;
    sparse_matrix_storage(const self&) = default;
    sparse_matrix_storage(self&&) = default;
    self& operator=(const self&) = default;
    self& operator=(self&&) = default;

    bool operator==(const self& other) const {
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

    bool operator!=(const self& other) const {
        return !(*this == other);
    }

    // Conditional Compilation for set_value and add_value
#ifdef DELETE_ZERO
    void set_value(size_type row, size_type column, const value_type& value) {
        std::vector<node_>& row_root = get_nth_row(row);
        bool iszero = value_equal(value, value_type(0));
        bool insert = false;
        bool update = false;
        for (auto it = row_root.begin(); it != row_root.end(); ++it) {
            if (it->column_ == column) {
                it->value_ = value;
                update = true;
                if (iszero) {
                    row_root.erase(it);
                    --element_count_;
                }
                break;
            }
            else if (it->column_ > column) {
                insert = true;
                break;
            }
        }
        if (update || iszero) {
            return;
        }
        else if (!update && !insert) {
            row_root.emplace_back(column, value);
            ++element_count_;
            return;
        }
        else {
            row_root.emplace_back(column, value);
            // Sort by column index
            std::sort(row_root.begin(), row_root.end(),
                [](const node_& n1, const node_& n2) -> bool {
                    return n1.column_ < n2.column_;
                });
            ++element_count_;
            return;
        }
    }

    void add_value(size_type row, size_type column, const value_type& value) {
        bool iszero = value_equal(value, value_type(0));
        if (iszero)
            return;
        std::vector<node_>& row_root = get_nth_row(row);
        bool insert = false;
        bool update = false;
        for (auto it = row_root.begin(); it != row_root.end(); ++it) {
            if (it->column_ == column) {
                it->value_ += value;
                update = true;
                if (value_equal(it->value_, value_type(0))) {
                    row_root.erase(it);
                    --element_count_;
                }
                break;
            }
            else if (it->column_ > column) {
                insert = true;
                break;
            }
        }
        if (update) {
            return;
        }
        else if (!update && !insert) {
            row_root.emplace_back(column, value);
            ++element_count_;
            return;
        }
        else {
            row_root.emplace_back(column, value);
            // Sort by column index
            std::sort(row_root.begin(), row_root.end(),
                [](const node_& n1, const node_& n2) -> bool {
                    return n1.column_ < n2.column_;
                });
            ++element_count_;
            return;
        }
    }
#else
    // Standard set_value implementation without deleting zeros
    void set_value(size_type row, size_type column, const value_type& value) {
        std::vector<node_>& row_root = get_nth_row(row);
        auto it = std::find_if(row_root.begin(), row_root.end(),
            [column](const node_& node) -> bool {
                return node.column_ == column;
            });
        if (it == row_root.end()) {
            row_root.emplace_back(column, value);
            std::sort(row_root.begin(), row_root.end(),
                [](const node_& n1, const node_& n2) -> bool {
                    return n1.column_ < n2.column_;
                });
            ++element_count_;
        }
        else {
            it->value_ = value;
        }
    }

    // Standard add_value implementation without deleting zeros
    void add_value(size_type row, size_type column, const value_type& value) {
        std::vector<node_>& row_root = get_nth_row(row);
        auto it = std::find_if(row_root.begin(), row_root.end(),
            [column](const node_& node) -> bool {
                return node.column_ == column;
            });
        if (it == row_root.end()) {
            row_root.emplace_back(column, value);
            std::sort(row_root.begin(), row_root.end(),
                [](const node_& n1, const node_& n2) -> bool {
                    return n1.column_ < n2.column_;
                });
            ++element_count_;
        }
        else {
            it->value_ += value;
        }
    }
#endif // DELETE_ZERO

    value_type get_value(size_type row, size_type column) const {
        for (const auto& node : get_nth_row(row)) {
            if (node.column_ == column)
                return node.value_;
        }
        return value_type(0);
    }

    inline size_type get_row() const {
        return my_row_;
    }

    inline size_type get_column() const {
        return my_column_;
    }

    size_type get_nth_row_size(size_type row) const {
        return get_nth_row(row).size();
    }

    void delete_row(size_type row) {
        element_count_ -= get_nth_row_size(row);
        container_.erase(container_.begin() + row);
        --my_row_;
    }

    void delete_column(size_type column) {
        --my_column_;
        for (auto& each_row : container_) {
            auto& this_row = each_row.this_row_;
            for (auto it = this_row.begin(); it != this_row.end(); ) {
                if (it->column_ == column) {
                    it = this_row.erase(it);
                    --element_count_;
                }
                else {
                    if (it->column_ > column) {
                        it->column_ -= 1;
                    }
                    ++it;
                }
            }
        }
    }

    void resize(size_type new_row, size_type new_column) {
        assert(new_row > 0 && new_column > 0);
        if (new_row != my_row_) {
            if (new_row < my_row_) {
                for (size_type i = new_row; i < my_row_; ++i) {
                    element_count_ -= get_nth_row_size(i);
                }
            }
            container_.resize(new_row);
        }
        if (new_column < my_column_) {
            for (auto& row_iter : container_) {
                auto& this_row = row_iter.this_row_;
                for (auto it = this_row.begin(); it != this_row.end(); ) {
                    if (it->column_ >= new_column) {
                        it = this_row.erase(it);
                        --element_count_;
                    }
                    else {
                        ++it;
                    }
                }
            }
        }
        my_row_ = new_row;
        my_column_ = new_column;
    }

    void element_row_transform_swap(size_type row_i, size_type row_j) {
        std::swap(get_nth_row(row_i), get_nth_row(row_j));
    }

    void element_row_transform_multi(size_type row, value_type k) {
        std::vector<node_>& this_row = get_nth_row(row);
        for (auto& colu : this_row) {
            colu.value_ *= k;
        }
    }

    void element_row_transform_plus(size_type row_i, size_type row_j, value_type k) {
        std::vector<node_>& source_row = get_nth_row(row_j);
        for (const auto& colu : source_row) {
            add_value(row_i, colu.column_, colu.value_ * k);
        }
    }

    size_type get_element_count() const {
        return element_count_;
    }

    // Iterators
    class row_iterator {
    private:
        self* handle_;
        typename std::vector<each_row_container_>::iterator proxy_;
        size_type row_index_;

    public:
        row_iterator(self* h, typename std::vector<each_row_container_>::iterator it, size_type r)
            : handle_(h), proxy_(it), row_index_(r) {}

        row_iterator& operator++() {
            ++row_index_;
            ++proxy_;
            return *this;
        }

        row_iterator operator++(int) {
            row_iterator result = *this;
            ++(*this);
            return result;
        }

        bool operator==(const row_iterator& other) const {
            return proxy_ == other.proxy_;
        }

        bool operator!=(const row_iterator& other) const {
            return proxy_ != other.proxy_;
        }

        size_type row_index() const {
            return row_index_;
        }

        class column_iterator {
        private:
            typename std::vector<node_>::iterator proxy_;
            size_type row_;

        public:
            column_iterator(typename std::vector<node_>::iterator it, size_type r) : proxy_(it), row_(r) {}

            column_iterator& operator++() {
                ++proxy_;
                return *this;
            }

            column_iterator operator++(int) {
                column_iterator result = *this;
                ++(*this);
                return result;
            }

            bool operator==(const column_iterator& other) const {
                return proxy_ == other.proxy_;
            }

            bool operator!=(const column_iterator& other) const {
                return proxy_ != other.proxy_;
            }

            value_type& operator*() {
                return proxy_->value_;
            }

            value_type* operator->() {
                return &(proxy_->value_);
            }

            size_type column_index() const {
                return proxy_->column_;
            }

            size_type row_index() const {
                return row_;
            }
        };

        column_iterator begin() {
            return column_iterator(handle_->container_.at(row_index_).this_row_.begin(), row_index_);
        }

        column_iterator end() {
            return column_iterator(handle_->container_.at(row_index_).this_row_.end(), row_index_);
        }
    };

    class const_row_iterator {
    private:
        const self* const handle_;
        typename std::vector<each_row_container_>::const_iterator proxy_;
        size_type row_index_;

    public:
        const_row_iterator(const self* const h, typename std::vector<each_row_container_>::const_iterator it, size_type r)
            : handle_(h), proxy_(it), row_index_(r) {}

        const_row_iterator& operator++() {
            ++row_index_;
            ++proxy_;
            return *this;
        }

        const_row_iterator operator++(int) {
            const_row_iterator result = *this;
            ++(*this);
            return result;
        }

        bool operator==(const const_row_iterator& other) const {
            return proxy_ == other.proxy_;
        }

        bool operator!=(const const_row_iterator& other) const {
            return proxy_ != other.proxy_;
        }

        size_type row_index() const {
            return row_index_;
        }

        class const_column_iterator {
        private:
            typename std::vector<node_>::const_iterator proxy_;
            size_type row_;

        public:
            const_column_iterator(typename std::vector<node_>::const_iterator it, size_type r)
                : proxy_(it), row_(r) {}

            const_column_iterator& operator++() {
                ++proxy_;
                return *this;
            }

            const_column_iterator operator++(int) {
                const_column_iterator result = *this;
                ++(*this);
                return result;
            }

            bool operator==(const const_column_iterator& other) const {
                return proxy_ == other.proxy_;
            }

            bool operator!=(const const_column_iterator& other) const {
                return proxy_ != other.proxy_;
            }

            const value_type& operator*() const {
                return proxy_->value_;
            }

            const value_type* operator->() const {
                return &(proxy_->value_);
            }

            size_type column_index() const {
                return proxy_->column_;
            }

            size_type row_index() const {
                return row_;
            }
        };

        const_column_iterator begin() const {
            return const_column_iterator(handle_->container_.at(row_index_).this_row_.begin(), row_index_);
        }

        const_column_iterator end() const {
            return const_column_iterator(handle_->container_.at(row_index_).this_row_.end(), row_index_);
        }
    };

    row_iterator begin() {
        return row_iterator(this, container_.begin(), 0);
    }

    row_iterator end() {
        return row_iterator(this, container_.begin() + my_row_, my_row_);
    }

    const_row_iterator begin() const {
        return const_row_iterator(this, container_.begin(), 0);
    }

    const_row_iterator end() const {
        return const_row_iterator(this, container_.begin() + my_row_, my_row_);
    }

private:
    // Helper functions to access rows
    inline const std::vector<node_>& get_nth_row(size_type row) const {
        return container_.at(row).this_row_;
    }

    inline std::vector<node_>& get_nth_row(size_type row) {
        return container_.at(row).this_row_;
    }
};

} // namespace pnmatrix

#endif // MATRIX_STORAGE_CEP_H

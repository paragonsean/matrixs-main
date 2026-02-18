#pragma once

/**
 * @file matrix_storage_cep_config.h
 * @brief Configuration settings for sparse matrix storage (CEP format)
 * 
 * This file contains compile-time configuration options for the sparse matrix
 * storage implementation. The CEP (Compressed Element Pattern) storage format
 * is designed for efficient handling of sparse matrices with configurable
 * behavior for zero elements.
 * 
 * Configuration Options:
 * - DELETE_ZERO: Controls whether zero elements are automatically removed
 *                from the sparse storage when set or modified.
 *                When undefined (current setting), zero elements are retained
 *                in the storage structure, which may be useful for certain
 *                algorithms that temporarily store zero values.
 * 
 * Usage:
 * This file is included by sparse_matrix_storage.h and affects the behavior
 * of sparse matrix operations throughout the library.
 */

/**
 * @brief Zero element deletion control for sparse matrices
 * 
 * When DELETE_ZERO is defined:
 * - Setting an element to zero will remove it from storage
 * - Memory usage is optimized by eliminating zero entries
 * - Iterators skip over zero elements automatically
 * 
 * When DELETE_ZERO is undefined (current behavior):
 * - Zero elements are retained in the sparse structure
 * - May increase memory usage but preserves element positions
 * - Useful for algorithms that modify values frequently
 * 
 * To enable zero deletion: #define DELETE_ZERO
 * To disable zero deletion: #undef DELETE_ZERO (current)
 */
#undef DELETE_ZERO

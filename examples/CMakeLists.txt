set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Main example executable
add_executable(pnmatrix_example main.cpp examples.cpp)

# Matrix generation executable
add_executable(generate_matrix generateMatrix.cpp)

# Thread pool test executable
add_executable(thread_pool_test thread_pool_test.cpp thread_pool.h)

# Performance benchmark executable
add_executable(benchmark_example benchmark_example.cpp)

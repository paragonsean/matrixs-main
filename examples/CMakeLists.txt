cmake_minimum_required(VERSION 3.10)

# Include directories
include_directories(../include)
include_directories(../third_party)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Add definitions
add_definitions(-DUSE_CHRONO)

# Performance reporting executable (fixed version)
add_executable(performance_report_fixed performance_report_fixed.cpp)
target_link_libraries(performance_report_fixed PRIVATE)

# Main example executable
add_executable(pnmatrix_example main.cpp examples.cpp)

# Matrix generation executable
add_executable(generate_matrix generateMatrix.cpp)

# Thread pool test executable
add_executable(thread_pool_test thread_pool_test.cpp thread_pool.h)

# Performance benchmark executable
add_executable(benchmark_example benchmark_example.cpp)

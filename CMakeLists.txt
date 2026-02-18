cmake_minimum_required(VERSION 3.10)

project(pnmatrix C CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Set the output directories for the binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Option to build tests
option(MAKE_TESTS "if you want to make tests" ON)
if(MAKE_TESTS)
    add_subdirectory("${PROJECT_SOURCE_DIR}/test")
endif()

# Option to build examples
option(MAKE_EXAMPLES "if you want to make examples" ON)
if(MAKE_EXAMPLES)
    add_subdirectory("${PROJECT_SOURCE_DIR}/examples")
endif()

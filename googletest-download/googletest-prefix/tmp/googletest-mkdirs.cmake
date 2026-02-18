# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/spocams/Downloads/matrixs-main/googletest-src")
  file(MAKE_DIRECTORY "/Users/spocams/Downloads/matrixs-main/googletest-src")
endif()
file(MAKE_DIRECTORY
  "/Users/spocams/Downloads/matrixs-main/googletest-build"
  "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix"
  "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/tmp"
  "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/src/googletest-stamp"
  "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/src"
  "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/src/googletest-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/src/googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/spocams/Downloads/matrixs-main/googletest-download/googletest-prefix/src/googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()

# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.19)

find_package (Eigen3 3.3 REQUIRED)

set(LIBRARY_NAME BayesianFilters)
set(LIBRARY_FOLDER lib)

# Set project name and version
project(LIBRARY_NAME VERSION 0.1)

# Set environment variables
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Library examples
add_subdirectory(examples)

# Library unit tests
add_subdirectory(test)

cmake_minimum_required(VERSION 3.12)
project(iterative_focal)

find_package(Eigen3 QUIET)
if (EIGEN3_FOUND)
    message("Eigen found.")
    include_directories(${EIGEN3_INCLUDE_DIR})
else (EIGEN3_FOUND)
    message("Eigen not found.")
endif (EIGEN3_FOUND)

find_package(pybind11 REQUIRED)
pybind11_add_module(iterative_focal iterative_focal.cpp)
target_compile_options(iterative_focal PRIVATE "/O2")
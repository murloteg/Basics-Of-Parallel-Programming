cmake_minimum_required(VERSION 3.16.3)
project(lab5)

set(CMAKE_CXX_STANDARD 11)

add_executable(lab5 main.cpp)

find_package(MPI REQUIRED)
include_directories("${MPI_CXX_INCLUDE_DIRS}")
target_link_libraries(lab5 ${MPI_LIBRARIES})
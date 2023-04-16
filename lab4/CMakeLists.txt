cmake_minimum_required(VERSION 3.25)
project(lab4)

set(CMAKE_CXX_STANDARD 11)

add_executable(lab4 main.cpp)

find_package(MPI REQUIRED)

include_directories("${MPI_CXX_INCLUDE_DIRS}")

target_link_libraries(lab4 ${MPI_LIBRARIES})

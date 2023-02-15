cmake_minimum_required(VERSION 3.16.3)
project(lab1-version-2)

set(CMAKE_CXX_STANDARD 14)

add_executable(lab1-version-2 main.cpp)

find_package(MPI REQUIRED)

include_directories("${MPI_CXX_INCLUDE_DIRS}")

target_link_libraries(lab1-version-2 ${MPI_LIBRARIES})

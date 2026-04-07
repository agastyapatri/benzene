cmake_minimum_required(VERSION 3.3)
project(benzene)

add_executable(benzene main.cpp)
target_compile_features(benzene PRIVATE cxx_std_17)

find_package(OpenBLAS REQUIRED)
target_link_libraries(benzene PRIVATE OpenBLAS::OpenBLAS)

add_subdirectory(src)
target_include_directories(benzene PRIVATE src/)






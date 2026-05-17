cmake_minimum_required(VERSION 3.30)
project(benzene VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#finding OpenBLAS
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

#	Gathering source files 
set(SOURCES
	src/tensor.cpp 
	src/nn.cpp
	gpt2/gpt2.cpp
)

add_executable(benzene main.cpp ${SOURCES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(benzene PRIVATE -O0 -g)
else()
    target_compile_options(benzene PRIVATE -O3 -march=native)
endif()


#	include directories: this is crucial
target_include_directories(benzene PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(benzene PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpt2)

#	linking the math library
target_link_libraries(benzene PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)


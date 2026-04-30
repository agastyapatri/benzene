cmake_minimum_required(VERSION 3.30)
project(benzene VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#finding OpenBLAS
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)


#	Gathering source files and telling cmake to build all sources and main.cpp into a single file called benzene.
set(SOURCES
	src/tensor.cpp 
	src/nn.cpp
)
add_compile_options(-O3  -march=native -fopenmp)  
add_executable(benzene main.cpp ${SOURCES})

#	include directories: this is crucial
target_include_directories(benzene PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

#	linking the math library
target_link_libraries(benzene PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)


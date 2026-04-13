#include "tensor.hpp"
#include <numbers> 
#include <iostream> 
#include <chrono> 
#define ROWS 1024 * 10
#define COLS 1024 * 10
#define ITER 10

int main(){
	bz::tensor t1 = bz::tensor::randn({ROWS});
	bz::tensor t2 = bz::tensor::randn({ROWS});
	t1.at({50}) = 1000.000;
	std::cout << t1.at({50}) << std::endl; 
}

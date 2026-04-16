#include "tensor.hpp"
#include <iostream> 
#define ROWS 5
#define COLS 5
#define ITER 10

int main(){
	bz::tensor::manual_seed(0);
	bz::tensor t1 = bz::tensor::eye(ROWS);
	bz::tensor t2 = bz::tensor::randn({ROWS, COLS});
	// std::cout << (bz::matmul(t1, t2) == bz::matmul(t2, t1)) << std::endl;
}

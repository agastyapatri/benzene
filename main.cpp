#include "tensor.hpp"
#include <numbers> 
#include <iostream> 
#include <chrono> 
#define ROWS 5
#define COLS 5
#define ITER 10

int main(){
	bz::tensor t1 = bz::tensor::randn({ROWS, COLS});
	bz::tensor t2 = bz::tensor::randn({ROWS - 1});

}

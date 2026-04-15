#include "tensor.hpp"
#include <numbers> 
#include <iostream> 
#include <chrono> 
#define ROWS 5
#define COLS 5
#define ITER 10

#include <unordered_map>
int main(){
	bz::tensor::manual_seed(0);
	bz::tensor t1 = bz::tensor::randn({2, 5, 5});
	std::cout << t1 << std::endl;
	std::cout << std::endl;
	std::cout << bz::transpose(t1, 0, 1) << std::endl;






}

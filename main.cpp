#include "nn.hpp"
#include "tensor.hpp"
#include <chrono>
#include <iostream> 
#include <memory>
#include <string>
#define ROWS 1
#define COLS 784
#define ITER 10000
using tensor = bz::tensor;
namespace nn = bz::nn;

int main(){
	tensor t = tensor::randn({5,5,5});
	std::cout << t << std::endl << std::endl;
	tensor t2 = tensor::ones({5});
	std::cout << t + t2 << std::endl;






}


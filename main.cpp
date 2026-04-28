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
	tensor::manual_seed(0);
	tensor a = tensor::randn({10, 10}); //	weight tensor
	bz::vi32 idxs({1,2,3,4});
	std::cout << a << std::endl;
	std::cout << a.gather(idxs) << std::endl;






}


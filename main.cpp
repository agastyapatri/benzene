#include "nn.hpp"
#include "tensor.hpp"
#include <iostream> 
#define ROWS 5
#define COLS 5
#define ITER 10

int main(){
	bz::tensor::manual_seed(0);
	bz::tensor x =  bz::tensor::randn({3, 10});
	bz::nn::Linear l1(10, 1);
	bz::tensor out = l1(x);
	std::cout << bz::gelu(out) << std::endl;

}

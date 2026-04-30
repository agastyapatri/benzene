#include "tensor.hpp"
#define ROWS 1
#define COLS 784
#define ITER 10000
using tensor = bz::tensor;

int main(){
	tensor t1 = tensor::randn({10,11,12,13,14});
	tensor t2 = tensor::randn({10,11,12,14,15});
	tensor t3 = bz::matmul(t1, t2);
	for(auto i : t3.shape())
		std::cout << i << " ";
	std::cout << std::endl;
	std::cout << t2.numel() << std::endl;
}



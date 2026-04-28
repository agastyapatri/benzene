#include "tensor.hpp"
#define ROWS 1
#define COLS 784
#define ITER 10000
using tensor = bz::tensor;

int main(){
	tensor t1 = tensor::randn({5,5,5});
	tensor t2 = tensor::randn({5,2,3});
	tensor t3 = bz::matmul(t1, t2);
	std::cout << t3 << std::endl;





}


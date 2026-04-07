#include "tensor.hpp"
int main(){
	bz::tensor t1 = bz::tensor::zeros({2,3,4,5});
	bz::tensor t2 = bz::tensor::ones({2,3,4,5});
	float scalar = 3.14159;
	bz::tensor t3 = t2 / scalar;
	std::cout << t3 << std::endl;
}

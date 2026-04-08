#include "tensor.hpp"
int main(){
	bz::tensor t = bz::tensor::rand_uniform({2,3,4}, -1, 1);
	std::cout << t << std::endl;
}

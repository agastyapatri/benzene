#include "tensor.hpp"
#include <numbers> 
int main(){
	bz::tensor t = bz::tensor::zeros({2,3,4});
	t.fill(90 * (std::numbers::pi / 180));
	std::cout << t.cos() << std::endl;
}

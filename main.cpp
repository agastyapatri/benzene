#include "tensor.hpp"
#include <numbers> 
int main(){
	bz::tensor t = bz::tensor::randn({2,3,4});
	bz::tensor out = t.pow(3);
	std::cout << t << std::endl;
	std::cout << "\n";
	std::cout << out << std::endl;


}

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
	tensor labels = tensor::ones({ROWS, 1});

	nn::Sequential net;
	net.push_back(std::make_unique<nn::Linear>(784, 196));
	net.push_back(std::make_unique<nn::Linear>(196, 98));
	net.push_back(std::make_unique<nn::Linear>(98, 10));
	net.push_back(std::make_unique<nn::Linear>(10, 1));


	for(bz::u32 i = 0; i < net.num_layers(); i++){
		std::string wname = std::to_string(i) +  ".weight";
		std::string bname = std::to_string(i) +  ".bias";
		std::cout << wname << ": " << net.state_dict()[wname]->shape()[0] << " " <<   net.state_dict()[wname]->shape()[1] << std::endl;
		std::cout << bname << ": " << net.state_dict()[bname]->shape()[0] << " " <<   net.state_dict()[bname]->shape()[1] << std::endl;
	}




}


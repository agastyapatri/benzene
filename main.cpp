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
	tensor inputs = tensor::randn({ROWS, COLS});
	tensor labels = tensor::ones({ROWS, 1});

	nn::Sequential net;
	net.push_back(nn::make_linear(COLS, COLS/2));
	net.push_back(nn::make_gelu());
	net.push_back(nn::make_linear(COLS/2, COLS/4));
	net.push_back(nn::make_gelu());
	net.push_back(nn::make_linear(COLS/4, 10));
	net.push_back(nn::make_gelu());
	net.push_back(nn::make_linear(10, 1));
	net.push_back(nn::make_gelu());
	net.push_back(nn::make_softmax());


	tensor out = net(inputs);
	std::cout << out << std::endl;







}


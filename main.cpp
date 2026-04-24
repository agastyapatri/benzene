#include "nn.hpp"
#include "tensor.hpp"
#include <chrono>
#include <iostream> 
#include <memory>
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




	int samples = 0; 
	auto start = std::chrono::high_resolution_clock::now();
	while(samples <= 60000){
		tensor inputs = tensor::randn({ROWS, COLS});
		tensor out = net.forward(inputs);
		samples += ROWS;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "time taken to process the MNIST dataset for a single epoch: " << elapsed.count() * 1e-6 << " seconds" << std::endl;
	std::cout << samples << std::endl;
}


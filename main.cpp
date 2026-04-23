#include "nn.hpp"
#include "tensor.hpp"
#include <climits>
#include <iostream> 
#include <chrono> 
#define ROWS 10240
#define COLS 768
#define ITER 10000
using tensor = bz::tensor;
namespace nn = bz::nn;

int main(){
	tensor inputs = tensor::randn({ROWS, COLS});
	tensor ground_truth = tensor::ones({ROWS, 10});
	nn::Linear l1(768, 394);
	nn::Linear l2(394, 192);
	nn::Linear l3(192, 96);
	nn::Linear l4(96, 10);

	auto start = std::chrono::high_resolution_clock::now();
	tensor out = inputs;
	out = l1(out);
	out = l2(out);
	out = l3(out);
	out = l4(out);
	out = bz::gelu(out);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << (double)elapsed.count() * 1e-6 << std::endl;
	for(auto i : out.shape())
		std::cout << i << " ";
	std::cout << std::endl;
	












	return 0;
}

#include "nn.hpp"
#include "tensor.hpp"
#include <iostream> 
#include <chrono> 
#define ROWS 25*1024
#define COLS 5*1024
#define ITER 10000

int main(){

	bz::tensor t1 = bz::tensor::ones({ROWS});
	bz::tensor t2 = bz::tensor::ones({ROWS});


	double time = 0;
	for(int i = 0; i < ITER; i++){
		auto start = std::chrono::high_resolution_clock::now(); 
		bz::gelu(t1);
		// t1.sigmoid_();
		auto end = std::chrono::high_resolution_clock::now(); 
		auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		
		time += elapsed.count();

	}
	std::cout << time/ITER << std::endl;
	std::cout << t1.at({ROWS - 1}) << std::endl;
	return 0;
}

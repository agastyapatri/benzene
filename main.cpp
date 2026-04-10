#include "tensor.hpp"
#include <numbers> 
#include <iostream> 
#include <chrono> 
#define ROWS 10
#define COLS 10
#define ITER 10

int main(){
	bz::tensor t1 = bz::tensor::randn({ROWS, COLS});
	bz::tensor t2 = bz::tensor::randn({COLS});
	

	auto start = std::chrono::high_resolution_clock::now();
	bz::tensor out;
	for(int i = 0; i < ITER; i++){
		out = bz::tensor::mat_vec_mul(t1, t2);
	}
	std::cout << out << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = (end - start) / ITER;
	double ops = 2.0 * ROWS * COLS * COLS;
	double gflops = (ops / elapsed.count()) / 1e9; 
	std::cout << "Time taken for matmul with matrices of size " << ROWS << " x " << COLS << ": " << elapsed.count() << " seconds\n";
	std::cout << "Performance: " << gflops << "GFLOPS" << std::endl;

}
